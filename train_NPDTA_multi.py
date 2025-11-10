import os
import numpy as np
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from NPDTA_mse import Trainer
from NPDTA_training_multi import training
from utils import helper
from utils.loader import Preprocess
from utils.drug_aux import get_aux_drug
import nni


params = {
    'data_dir': 'data/Davis',
    'first_embedding_dim': 64,
    'second_embedding_dim': 16,
    'z_dim': 32,
    'e_dim': 256,
    'l_dim': 128,
    'lr': 0.0005,
    'num_epoch': 4000,
    'support_and_context': 30,
    'query_size': 40,
    'top_n': 5,
    'top_m': 7,
    'clusters_k': 11,
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=params['data_dir'])
parser.add_argument('--model_save_dir', type=str, default='save_model_dir')
parser.add_argument('--id', type=str, default='1', help='used for save hyper-parameters.')

parser.add_argument('--first_embedding_dim', type=int, default=params['first_embedding_dim'], help='Embedding dimension.')
parser.add_argument('--second_embedding_dim', type=int, default=params['second_embedding_dim'], help='Embedding dimension.')
parser.add_argument('--top_n', type=int, default=params['top_n'], help='Number of auxiliary drugs.')
parser.add_argument('--top_m', type=int, default=params['top_m'], help='Number of historical interaction targets.')

parser.add_argument('--z1_dim', type=int, default=params['z_dim'], help='The dimension of r.')
parser.add_argument('--z2_dim', type=int, default=params['z_dim'], help='The dimension of hidden.')
parser.add_argument('--z_dim', type=int, default=params['z_dim'], help='The dimension of z.')

parser.add_argument('--e_h1_dim', type=int, default=params['e_dim'], help='The hidden first dimension of experts.')
parser.add_argument('--e_h2_dim', type=int, default=params['e_dim'], help='The hidden second dimension of experts.')
parser.add_argument('--top_e', type=int, default=3, help='The number of experts selected.')
parser.add_argument('--num_e', type=int, default=6, help='The total number of experts in MoE.')

parser.add_argument('--h_h1_dim', type=int, default=128, help='The hidden first dimension of PML.')
parser.add_argument('--h_h2_dim', type=int, default=64, help='The hidden second dimension of PML.')
parser.add_argument('--h_final_dim', type=int, default=64, help='The hidden third dimension of PML.')
parser.add_argument('--clusters_k', type=int, default=params['clusters_k'], help='Cluster number of prototypes.')
parser.add_argument('--temperature', type=float, default=1.0, help='used for student-t distribution.')
parser.add_argument('--lambda', type=float, default=0.1, help='used to balance the clustering loss and NP loss.')

parser.add_argument('--l_h1_dim', type=int, default=params['l_dim'], help='The hidden first dimension of multi-scale fusion.')
parser.add_argument('--l_h2_dim', type=int, default=params['l_dim'], help='The hidden second dimension of multi-scale fusion.')
parser.add_argument('--l_h3_dim', type=int, default=params['l_dim'], help='The hidden third dimension of multi-scale fusion.')

parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--lr', type=float, default=params['lr'], help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=params['num_epoch'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio for training.')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='ratio for validation.')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--support_size', type=int, default=params['support_and_context'])
parser.add_argument('--query_size', type=int, default=params['query_size'])
parser.add_argument('--max_len', type=int, default=500, help='The max number of interactions for each drug.')
parser.add_argument('--context_min', type=int, default=params['support_and_context'], help='Minimum size of context range.')

args = parser.parse_args()


def seed_everything(seed=2020):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


seed = args.seed
seed_everything(seed)

if args.cpu:
    args.use_cuda = False
elif args.use_cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

# print model info
helper.print_config(opt)
helper.ensure_dir(opt["model_save_dir"], verbose=True)
# save model config
helper.save_config(opt, opt["model_save_dir"] + "/" + opt["id"] + '.config', verbose=True)
# record training log
file_logger = helper.FileLogger(opt["model_save_dir"] + '/' + opt['id'] + ".log",
                                header="# epoch\ttrain_loss\tprecision5\tNDCG5\tMAP5\tprecision7"
                                       "\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")

#
# preprocess = Preprocess(opt)
# get_aux_drug(opt["data_dir"], "training")
# get_aux_drug(opt["data_dir"], "testing")
print("Preprocess is done.")
print("Create model NPDTA...")

opt['df_dim'] = 384
opt['tf_dim'] = 320

model_filename = "{}/{}.pt".format(opt['model_save_dir'], opt["id"])

training_set_size = int(len(os.listdir("{}/{}/{}".format(opt["data_dir"], "training", "log"))) / 4)
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []
for idx in range(training_set_size):
    supp_xs_s.append(pickle.load(open("{}/{}/{}/supp_x_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    supp_ys_s.append(pickle.load(open("{}/{}/{}/supp_y_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    query_xs_s.append(pickle.load(open("{}/{}/{}/query_x_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    query_ys_s.append(pickle.load(open("{}/{}/{}/query_y_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
train_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

testing_set_size = int(len(os.listdir("{}/{}/{}".format(opt["data_dir"], "testing", "log"))) / 4)
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []
for idx in range(testing_set_size):
    supp_xs_s.append(
        pickle.load(open("{}/{}/{}/supp_x_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    supp_ys_s.append(
        pickle.load(open("{}/{}/{}/supp_y_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    query_xs_s.append(
        pickle.load(open("{}/{}/{}/query_x_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    query_ys_s.append(
        pickle.load(open("{}/{}/{}/query_y_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
test_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

if not os.path.exists(model_filename):
    print("Start training...")
    training(opt, train_dataset, test_dataset, batch_size=opt['batch_size'], num_epoch=opt['num_epoch'],
             model_save=opt["save"], model_filename=model_filename, logger=file_logger)

else:
    print("Load pre-trained model...")
    trainer = Trainer(opt)
    if opt['use_cuda']:
        trainer.cuda()
    opt = helper.load_config(model_filename[:-2]+"config")
    helper.print_config(opt)
    trained_state_dict = torch.load(model_filename)
    trainer.load_state_dict(trained_state_dict)
