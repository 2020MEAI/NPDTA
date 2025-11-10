import os
import torch
import pickle
import random
import numpy as np
from NPDTA_mse import Trainer
from eval import test_model
import nni


def calc_mean_std(values):
    values = np.array(values)
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1)
    }


def training(opt, train_dataset, test_dataset, batch_size, num_epoch, model_save=True, model_filename=None, logger=None):
    store_file = opt['data_dir'] + '/drug_similarity_training.pkl'
    with open(store_file, 'rb') as f:
        train_dict = pickle.load(f)
    store_file = opt['data_dir'] + '/drug_similarity_testing.pkl'
    with open(store_file, 'rb') as f:
        test_dict = pickle.load(f)
    result_dir = 'results.txt'

    all_best_metrics = []
    for repeat in range(10):
        trainer = Trainer(opt)
        if opt['use_cuda']:
            trainer.cuda()

        training_set_size = len(train_dataset)
        best_mse = 50
        best_metrics = None
        for epoch in range(num_epoch):
            random.shuffle(train_dataset)
            num_batch = int(training_set_size / batch_size)
            a, b, c, d = zip(*train_dataset)
            trainer.train()
            for i in range(num_batch):
                try:
                    supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                    supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                    query_xs = list(c[batch_size*i:batch_size*(i+1)])
                    query_ys = list(d[batch_size*i:batch_size*(i+1)])
                except IndexError:
                    continue
                train_loss = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys, train_dict)

            mloss, mmse, mmae, mr2, mrm2, mps, mci = test_model(trainer, opt, test_dataset, test_dict)
            logger.log("{}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
                epoch, mloss, mmse, mmae, mr2, mrm2, mps, mci))
            nni.report_intermediate_result(mmse)
            if mmse <= best_mse:
                best_mse = mmse
                best_metrics = {
                    'mse': mmse,
                    'mae': mmae,
                    'r2': mr2,
                    'rm2': mrm2,
                    'ps': mps,
                    'ci': mci
                }
        all_best_metrics.append(best_metrics)

        if model_save:
            torch.save(trainer.state_dict(), model_filename)

        with open(result_dir, 'a') as f:
            f.write(f"{best_metrics['mse']:.4f}\t{best_metrics['mae']:.4f}\t{best_metrics['r2']:.4f}"
                    f"\t{best_metrics['rm2']:.4f}\t{best_metrics['ps']:.4f}\t{best_metrics['ci']:.4f}\n")

    stats = {
        'MSE': calc_mean_std([m['mse'] for m in all_best_metrics]),
        'MAE': calc_mean_std([m['mae'] for m in all_best_metrics]),
        'R2': calc_mean_std([m['r2'] for m in all_best_metrics]),
        'RM2': calc_mean_std([m['rm2'] for m in all_best_metrics]),
        'PS': calc_mean_std([m['ps'] for m in all_best_metrics]),
        'CI': calc_mean_std([m['ci'] for m in all_best_metrics]),
    }

    with open(result_dir, 'a') as f:
        f.write("\n=== Mean and StdDev of Best Metrics ===\n")
        f.write("Metric\tMean(StdDev)\n")
        for key, stat in stats.items():
            f.write(f"{key}\t{stat['mean']:.4f}({stat['std']:.4f})\n")

    nni.report_final_result(float(stats['MSE']['mean']))
