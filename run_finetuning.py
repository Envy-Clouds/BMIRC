import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import torch.utils.data
from functools import partial

import utils
from models.input_adapters import PatchedInputAdapter
from models.output_adapters import LinearOutputAdapter
from utils import LabelSmoothingCrossEntropy, Mixup, ModelEma
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import SoftTargetCrossEntropy, create_model
from utils.datasets import build_dataset_ecg
from utils.optim_factory import (LayerDecayValueAssigner, create_optimizer)
from utils.evaluation import compute_accuracy, compute_AUC, calc_f1, calc_recall, compute_auprc

DOMAIN_CONF = {
    'ecg_t': {
        'channels': 12,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=12),
    },
    'ecg_f': {
        'channels': 12,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=12),
    },
}


def evaluate_metrics(y_true, y_pred):
    acc = compute_accuracy(y_true, y_pred)
    # auc = compute_AUC(y_true, y_pred)
    f1 = calc_f1(y_true, y_pred)
    auprc = compute_auprc(y_true, y_pred)
    rec = calc_recall(y_true, y_pred)
    return acc, f1, auprc, rec


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='cfgs/finetune/mymae.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MAE fine-tuning and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--in_domains', default='ecg-pcg', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')

    # Model parameters
    parser.add_argument('--model', default='multivit_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder')
    parser.add_argument('--patch_size', default=10, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--input_size', default=1000, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0., metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0., metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA')

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic) (default: "bicubic")')

    # Mixup parameters
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', default=False, action='store_true')
    parser.add_argument('--no_mean_pooling', action='store_false', dest='use_mean_pooling')
    parser.set_defaults(use_mean_pooling=True)

    # Dataset parameters
    parser.add_argument('--data_path', default='dataset/ecg/ningbo.npy', type=str, help='ecg_data path')
    parser.add_argument('--labels_path', default='dataset/ecgpcg/labels_5000.npy', type=str, help='labels path')
    parser.add_argument('--nb_classes', default=6, type=int, help='number of the classification types')
    parser.add_argument('--folds', default=4, type=int, help='number of the dataset folds')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                        help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Run name on wandb')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    return parser.parse_args(remaining)


def main(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    dataset_train, dataset_val, weights = build_dataset_ecg(args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    input_adapters = {
        'ecg_t': DOMAIN_CONF['ecg_t']['input_adapter'](
            stride_level=DOMAIN_CONF['ecg_t']['stride_level'],
            patch_size_full=args.patch_size,
            image_size=args.input_size
        ),
        'ecg_f': DOMAIN_CONF['ecg_f']['input_adapter'](
            stride_level=DOMAIN_CONF['ecg_f']['stride_level'],
            patch_size_full=args.patch_size,
            image_size=int(0.5*args.input_size)
        )
    }

    output_adapters = {
        'cls': LinearOutputAdapter(
            num_classes=args.nb_classes,
            use_mean_pooling=args.use_mean_pooling,
        )
    }

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu')
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            # for name, p in model.named_parameters():
            #     if name.startswith("output_adapters"):
            #         p.requires_grad = True
            #     else:
            #         p.requires_grad = False

        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.blr
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(weight=weights.to(device))

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        val_stats = evaluate(data_loader_val, model, device, args.in_domains)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_prc, best_epoch_prc = 0., 0.
    current_acc = 0.
    current_f1 = 0.
    current_auc = 0.
    current_rec = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            in_domains=args.in_domains
        )
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, args.in_domains)
            if max_prc < val_stats["prc"]:
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_prc = val_stats["prc"]
                best_epoch_prc = epoch
                current_acc = val_stats["acc"]
                current_f1 = val_stats["f1"]
                current_auc = val_stats["auc"]
                current_rec = val_stats["rec"]

            print(f'Best: Acc {current_acc:.5f}', f'Auc {current_auc:.5f}', f'F1 {current_f1:.5f}',
                  f'Prc {max_prc:.5f}', f'Rec {current_rec:.5f}')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         # **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        evals = {'seed': args.seed, 'epochs': args.epochs,
                 'prc_epoch': best_epoch_prc,
                 'current_acc': current_acc, 'current_auc': current_auc, 'current_f1': current_f1,
                 'max_prc': max_prc,
                 'current_rec': current_rec
                 }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "best.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(evals) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    outputs = outputs['cls']
    cls_loss = criterion(outputs, target)
    loss = {'cls': cls_loss}
    return loss, outputs


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, in_domains: List[str] = []):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    optimizer.zero_grad()
    outputs = []
    targets_list = []
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in samples.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            ori_losses, output = train_class_batch(
                model, input_dict, targets, criterion)
            loss = sum(ori_losses.values())

        loss_value = sum(ori_losses.values()).item()
        sp_loss_values = {f'{task}_loss': l for task, l in ori_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        output = torch.sigmoid(output)
        acc, f1, prc, rec = evaluate_metrics(targets.detach().cpu().numpy(), output.detach().cpu().numpy())

        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets_list.append(targets[i].cpu().detach().numpy())

        metric_logger.update(loss=loss_value)
        metric_logger.update(**sp_loss_values)
        metric_logger.meters['acc'].update(acc)
        metric_logger.meters['f1'].update(f1)
        metric_logger.meters['prc'].update(prc)
        metric_logger.meters['recall'].update(rec)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()

    acc_mean, f1_mean, prc_mean, rec_mean = evaluate_metrics(targets_list, outputs)
    auc_mean = compute_AUC(targets_list, outputs)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('Train: Acc {:.5f} Auc {:.5f} F1 {:.5f} Prc {:.5f} Rec {:.5f}'.format(acc_mean, auc_mean, f1_mean, prc_mean,
                                                                                rec_mean))
    return {'loss': metric_logger.loss.global_avg, 'acc': acc_mean, 'auc': auc_mean, 'f1': f1_mean, 'prc': prc_mean,
            'rec': rec_mean}


@torch.no_grad()
def evaluate(data_loader, model, device, in_domains: List[str] = []):
    outputs = []
    targets = []

    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 20, header):
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in batch[0].items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        target = batch[1]
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(input_dict)
            output = output['cls']
            cls_loss = criterion(output, target)
            ori_losses = {'cls': cls_loss}

        loss_value = sum(ori_losses.values()).item()
        sp_loss_values = {f'{task}_loss': l for task, l in ori_losses.items()}

        output = torch.sigmoid(output)
        acc, f1, prc, rec = evaluate_metrics(target.cpu().detach().numpy(), output.detach().cpu().numpy())

        batch_size = target.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(**sp_loss_values)
        metric_logger.meters['acc'].update(acc, n=batch_size)
        metric_logger.meters['f1'].update(f1, n=batch_size)
        metric_logger.meters['prc'].update(prc, n=batch_size)
        metric_logger.meters['rec'].update(rec, n=batch_size)

        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    acc_mean, f1_mean, prc_mean, rec_mean = evaluate_metrics(targets, outputs)
    auc_mean = compute_AUC(targets, outputs)

    print('Val loss {losses.global_avg:.5f}'.format(losses=metric_logger.loss))
    print('VAL: Acc {:.5f} Auc {:.5f} F1 {:.5f} Prc {:.5f} Rec {:.5f}'.format(acc_mean, auc_mean, f1_mean, prc_mean,
                                                                                rec_mean))
    return {'loss': metric_logger.loss.global_avg, 'acc': acc_mean, 'auc': auc_mean, 'f1': f1_mean, 'prc': prc_mean,
            'rec': rec_mean}


if __name__ == '__main__':
    opts = get_args()
    opts.model = 'my_backbone'
    opts.data_path = 'dataset/ecg/ningbo.npy'
    opts.labels_path = 'dataset/ecg/ningbo_label.npy'
    opts.nb_classes = 25
    opts.finetune = ''
    opts.output_dir = 'output/finetune/ningbo/mymae'
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
