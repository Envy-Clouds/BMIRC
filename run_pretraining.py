import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import torch.utils.data

import utils
from models.criterion import MaskedMSELoss
from models.input_adapters import PatchedInputAdapter
from models.output_adapters import MyOutputAdapter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.datasets import build_dataset_ecg
from utils.optim_factory import create_optimizer
from utils.task_balancing import (NoWeightingStrategy,
                                  UncertaintyWeightingStrategy)

DOMAIN_CONF = {
    'ecg_t': {
        'channels': 12,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=12),
        'output_adapter': partial(MyOutputAdapter, num_channels=12),
        'loss': MaskedMSELoss,
    },
    'ecg_f': {
        'channels': 12,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=12),
        'output_adapter': partial(MyOutputAdapter, num_channels=12),
        'loss': MaskedMSELoss,
    },
}


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='cfgs/pretrain/mymae.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--epochs', default=300, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--save_ckpt_freq', default=50, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')

    # Task parameters
    parser.add_argument('--in_domains', default='ecg_t-ecg_f', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='ecg_t-ecg_f', type=str,
                        help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)
    parser.add_argument('--extra_norm_pix_loss', action='store_true')
    parser.add_argument('--no_extra_norm_pix_loss', action='store_false', dest='extra_norm_pix_loss')
    parser.set_defaults(extra_norm_pix_loss=True)

    # Model parameters
    parser.add_argument('--model', default='pretrain_mymae', type=str, metavar='MODEL',
                        help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=50, type=int,
                        help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens_f', default=12, type=int,
                        help='Number of tokens to randomly choose for encoder_f (default: %(default)s)')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--patch_size', default=10, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=1000, type=int,
                        help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0,
                        help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                        help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true',
                        help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true',
                        help='Set to True/False to enable/disable decoder cross attention.')
    parser.add_argument('--decoder_dim', default=96, type=int,
                        help='Token dimension inside the decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
    parser.add_argument('--decoder_num_heads', default=3, type=int,
                        help='Number of attention heads in decoder (default: %(default)s)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: %(default)s)')

    parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                        help='Set to True/False to enable/disable computing the loss on non-masked tokens')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)
    parser.add_argument('--loss_weight_t', default=1.0, type=float, help='')
    parser.add_argument('--loss_weight_f', default=1.0, type=float, help='')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM',
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM',
                        help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.  (Set the same value as args.weight_decay to keep weight decay unchanged)""")
    parser.add_argument('--decoder_decay', type=float, default=None, help='decoder weight decay')

    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
    parser.add_argument('--task_balancer', type=str, default='none',
                        help='Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0,
                        help='Task loss balancer LR scale (if used) (default: %(default)s)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')

    # Augmentation parameters
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic) (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--data_path', default='dataset/ecg/ningbo.npy', type=str, help='ecg_data path')
    parser.add_argument('--labels_path', default='dataset/ecg/ningbo_label.npy', type=str, help='labels path')
    parser.add_argument('--folds', default=4, type=int, help='number of the dataset folds')

    # Misc.
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')

    parser.add_argument('--seed', default=42, type=int, help='Random seed ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

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

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def get_model(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

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
        'ecg_t': DOMAIN_CONF['ecg_t']['output_adapter'](
            stride_level=DOMAIN_CONF['ecg_t']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            image_size=args.input_size
        ),
        'ecg_f': DOMAIN_CONF['ecg_f']['output_adapter'](
            stride_level=DOMAIN_CONF['ecg_f']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            image_size=int(0.5*args.input_size)
        )
    }

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    )

    return model


def main(args):
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)

    if args.task_balancer == 'uncertainty':
        loss_balancer = UncertaintyWeightingStrategy(tasks=args.out_domains)
    else:
        loss_balancer = NoWeightingStrategy()

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](patch_size=args.patch_size, stride=DOMAIN_CONF[domain]['stride_level'])
        for domain in args.out_domains
    }

    # Get dataset
    dataset_train, _, _ = build_dataset_ecg(args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)  # 打乱数据集

    if args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size
    # 样本数//batch = n。需要n个batch循环完样本，一个epoch需要训练n次

    model.to(device)
    loss_balancer.to(device)
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size
    args.lr = args.blr

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(
        args, {'model': model_without_ddp, 'balancer': loss_balancer_without_ddp})
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            loss_balancer=loss_balancer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            num_encoded_tokens_f=args.num_encoded_tokens_f,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            loss_weight_t=args.loss_weight_t,
            loss_weight_f=args.loss_weight_f,
        )
        if log_writer is not None:
            log_writer.update({**{k: v for k, v in train_stats.items()}, 'epoch': epoch})
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, loss_balancer=loss_balancer_without_ddp, epoch=epoch)

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, tasks_loss_fn: Dict[str, torch.nn.Module],
                    loss_balancer: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = None, max_skip_norm: float = None,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_encoded_tokens: int = 196, num_encoded_tokens_f: int = 12, in_domains: List[str] = [],
                    loss_on_unmasked: bool = True,
                    loss_weight_t: float = 1.0, loss_weight_f: float = 1.0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with torch.cuda.amp.autocast():
            preds, masks = model(
                input_dict,
                num_encoded_tokens=num_encoded_tokens,
                num_encoded_tokens_f=num_encoded_tokens_f
            )

            task_losses = {}
            for task in preds:
                target = tasks_dict[task]
                if loss_on_unmasked:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))

            task_losses['ecg_t'] = loss_weight_t * task_losses['ecg_t']
            task_losses['ecg_f'] = loss_weight_f * task_losses['ecg_f']

            weighted_task_losses = loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_task_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
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
            log_writer.update(task_loss_values)
            log_writer.update(weighted_task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    opts = get_args()
    opts.model = 'pretrain_mymae'
    opts.data_path = 'dataset/ecg/ningbo.npy'
    opts.labels_path = 'dataset/ecg/ningbo_label.npy'
    opts.output_dir = 'output/pretrain/ningbo/mymae'
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
