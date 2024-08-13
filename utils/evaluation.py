# -*- coding: utf-8 -*-
'''
@time: 2021/4/7 15:21

@ author:
'''

import time
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, auc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1

    return sum / count


def compute_AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auc = []
    for i in range(len(y_true[0])):
        class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = np.mean(class_auc)
    return auc


def compute_auprc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auprc = []
    for i in range(len(y_true[0])):
        class_auprc.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    auprc = np.mean(class_auprc)
    return auprc


def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    y_pred_binary = apply_thresholds(y_pre, threshold)
    return f1_score(y_true, y_pred_binary, average="samples")


def calc_recall(y_true, y_pre, threshold=0.5):
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    y_pred_binary = apply_thresholds(y_pre, threshold)
    return recall_score(y_true, y_pred_binary, average="samples")


def apply_thresholds(preds, threshold):
    tmp = []
    for p in preds:
        tmp_p = (p > threshold).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp


# PRINT TIME
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)


# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    labels = np.array(labels)
    outputs = np.array(outputs)
    outputs = apply_thresholds(outputs, 0.5)
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


def AP(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num_instance, num_class = y_pred.shape
    precision_value = 0
    precisions = []
    for i in range(num_instance):
        p = precision_score(y_true[i, :], y_pred[i, :])
        precisions.append(p)
        precision_value += p
        # print(precision_value)
    pre_list = np.array([1.0] + precisions + [0.0])  # for get AUPRC
    # print(pre_list)
    return float(precision_value / num_instance), pre_list


def AR(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num_instance, num_class = y_pred.shape
    recall_value = 0
    recalls = []
    for i in range(num_instance):
        p = recall_score(y_true[i, :], y_pred[i, :])
        recalls.append(p)
        recall_value += p
    rec_list = np.array([0.0] + recalls + [1.0])  # for get AUPRC
    sorting_indices = np.argsort(rec_list)
    # print(rec_list)
    return float(recall_value / num_instance), rec_list, sorting_indices


def get_auprc(y_true, y_pred):
    avgPrecision, precisions = AP(y_true, y_pred)
    avfRecall, recalls, sorting_indices = AR(y_true, y_pred)
    # x is either increasing or decreasing
    # such as recalls[sorting_indices]
    auprc = auc(recalls[sorting_indices], precisions[sorting_indices])
    return auprc
