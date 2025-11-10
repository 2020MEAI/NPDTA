"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress, pearsonr, spearmanr
from lifelines.utils import concordance_index
import math


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def calculate_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = spearmanr(y, f)[0]
    return rs


def test_model(trainer, opt, test_dataset, test_dict):
    test_dataset_len = len(test_dataset)
    # batch_size = opt["batch_size"]
    minibatch_size = 1
    a, b, c, d = zip(*test_dataset)
    trainer.eval()

    all_loss = 0
    loss_count = 0
    all_preds = []
    all_labels = []
    for i in range(test_dataset_len):
        try:
            supp_xs = list(a[minibatch_size * i:minibatch_size * (i + 1)])
            supp_ys = list(b[minibatch_size * i:minibatch_size * (i + 1)])
            query_xs = list(c[minibatch_size * i:minibatch_size * (i + 1)])
            query_ys = list(d[minibatch_size * i:minibatch_size * (i + 1)])
        except IndexError:
            continue
        test_loss, total_preds, total_labels = trainer.query_rec(supp_xs, supp_ys, query_xs, query_ys, test_dict)
        all_loss += test_loss
        loss_count += 1

        all_preds = np.concatenate((all_preds, total_preds))
        all_labels = np.concatenate((all_labels, total_labels))

    loss = all_loss / loss_count
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    rm2 = calculate_rm2(all_labels, all_preds)
    ps, _ = pearsonr(all_labels, all_preds)
    ci = concordance_index(all_labels, all_preds)

    return loss, mse, mae, r2, rm2, ps, ci
