import json
import random
import torch
import numpy as np
import pickle
import codecs
import re
import os
import pandas as pd

# ----------------------------------------------------------------------------
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=900):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def collate_fn(list_d, list_p, max_d=100, max_p=900):
    dict_d = {}
    for i, compoundstr in enumerate(list_d):
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, max_d)).unsqueeze(0)
        dict_d[compoundstr] = compoundint

    dict_p = {}
    for i, proteinstr in enumerate(list_p):
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, max_p)).unsqueeze(0)
        dict_p[proteinstr] = proteinint

    return dict_d, dict_p
# ----------------------------------------------------------------------------


def to_onehot_dict(list):
    dict={}
    length = len(list)
    for index, element in enumerate(list):
        vector = torch.zeros(1, length).long()
        element = int(element)
        vector[:, element] = 1.0
        dict[element] = vector
    return dict


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def merge_key(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def merge_value(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1.keys():
            new_value = dict1[key]+value
            dict1[key] = new_value
        else:
            print('Unexpected key.')


def count_values(dict):
    count_val = 0
    for key, value in dict.items():
        count_val += len(value)
    return count_val


def construct_dictionary(drug_list, total_dict):
    dict = {}
    for i in range(len(drug_list)):
        dict[str(drug_list[i])] = total_dict[str(drug_list[i])]
    return dict


class Preprocess(object):
    """
    Preprocess the training, validation and testing data.
    Generate the episode-style data.
    """

    def __init__(self, opt):
        self.batch_size = opt["batch_size"]
        self.opt = opt

        self.train_ratio = opt['train_ratio']
        self.valid_ratio = opt['valid_ratio']
        self.test_ratio = 1 - self.train_ratio - self.valid_ratio
        self.dataset_path = opt["data_dir"]
        self.support_size = opt['support_size']
        self.query_size = opt['query_size']
        self.max_len = opt['max_len']
        df_dim, tf_dim = self.preprocess(self.dataset_path)
        self.df_dim = df_dim
        self.tf_dim = tf_dim

    def preprocess(self, dataset_path):
        """ Preprocess the data and convert to ids. """
        print('Create training, validation and testing data from scratch!')
        with open('./{}/interaction_dict_x.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            inter_dict_x = json.loads(f.read())
        with open('./{}/interaction_dict_y.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            inter_dict_y = json.loads(f.read())
        print('The size of total interactions is %d.' % (count_values(inter_dict_x)))
        assert count_values(inter_dict_x) == count_values(inter_dict_y)

        with open('./{}/drug_list.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            drugids = json.loads(f.read())

        with open('./{}/target_list.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            targetids = json.loads(f.read())

        random.shuffle(drugids)
        train_drug_size = int(len(drugids) * self.train_ratio)
        valid_drug_size = int(len(drugids) * self.valid_ratio)
        train_drugs = drugids[:train_drug_size]
        valid_drugs = drugids[train_drug_size: train_drug_size + valid_drug_size]
        test_drugs = drugids[train_drug_size + valid_drug_size:]
        assert len(drugids) == len(train_drugs) + len(valid_drugs) + len(test_drugs)

        # Construct the training data dict
        train_dict_x = construct_dictionary(train_drugs, inter_dict_x)
        train_dict_y = construct_dictionary(train_drugs, inter_dict_y)

        target_set = set()
        for i in train_dict_x.values():
            i = set(i)
            target_set = target_set.union(i)

        with open('embedding_Davis/molecule_mapping_emb_ChemBERTa.pkl', 'rb') as file:
            drug_dict = pickle.load(file)
        with open('embedding_Davis/protein_mapping_emb_ESM-2-320.pkl', 'rb') as file:
            target_dict = pickle.load(file)

        valid_dict_x = construct_dictionary(valid_drugs, inter_dict_x)
        valid_dict_y = construct_dictionary(valid_drugs, inter_dict_y)
        assert count_values(valid_dict_x) == count_values(valid_dict_y)

        test_dict_x = construct_dictionary(test_drugs, inter_dict_x)
        test_dict_y = construct_dictionary(test_drugs, inter_dict_y)
        assert count_values(test_dict_x) == count_values(test_dict_y)

        print('Test data has %d interactions.' % (count_values(test_dict_x)))

        def generate_episodes(dict_x, dict_y, category, support_size, query_size, max_len, dir="log"):
            idx = 0
            if not os.path.exists("{}/{}/{}".format(dataset_path, category, dir)):
                os.makedirs("{}/{}/{}".format(dataset_path, category, dir))
                os.makedirs("{}/{}/{}".format(dataset_path, category, "evidence"))
                for _, drug_id in enumerate(dict_x.keys()):
                    d_id = int(drug_id)
                    seen_music_len = len(dict_x[drug_id])
                    indices = list(range(seen_music_len))
                    # filter some drugs with their interactions, i.e., tasks
                    if seen_music_len < (support_size + query_size) or seen_music_len > max_len:
                        continue
                    random.shuffle(indices)
                    tmp_x = np.array(dict_x[drug_id])
                    tmp_y = np.array(dict_y[drug_id])

                    support_x_app = None
                    for t_id in tmp_x[indices[:support_size]]:
                        t_emb = torch.mean(target_dict[t_id], dim=0, keepdim=True)
                        d_emb = torch.mean(drug_dict[d_id], dim=0, keepdim=True)
                        tmp_x_converted = torch.cat((t_emb, d_emb), 1)
                        try:
                            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                        except:
                            support_x_app = tmp_x_converted

                    query_x_app = None
                    for t_id in tmp_x[indices[support_size:]]:
                        d_id = int(drug_id)
                        t_emb = torch.mean(target_dict[t_id], dim=0, keepdim=True)
                        d_emb = torch.mean(drug_dict[d_id], dim=0, keepdim=True)
                        tmp_x_converted = torch.cat((t_emb, d_emb), 1)
                        try:
                            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                        except:
                            query_x_app = tmp_x_converted

                    support_y_app = torch.FloatTensor(tmp_y[indices[:support_size]])
                    query_y_app = torch.FloatTensor(tmp_y[indices[support_size:]])

                    pickle.dump(support_x_app, open("{}/{}/{}/supp_x_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(support_y_app, open("{}/{}/{}/supp_y_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(query_x_app, open("{}/{}/{}/query_x_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(query_y_app, open("{}/{}/{}/query_y_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    # used for evidence candidate selection
                    with open("{}/{}/{}/supp_x_{}_ids.txt".format(dataset_path, category, "evidence", idx), "w") as f:
                        for t_id in tmp_x[indices[:support_size]]:
                            f.write("{}\t{}\n".format(drug_id, t_id))
                    with open("{}/{}/{}/query_x_{}_ids.txt".format(dataset_path, category, "evidence", idx), "w") as f:
                        for t_id in tmp_x[indices[support_size:]]:
                            f.write("{}\t{}\n".format(drug_id, t_id))
                    idx += 1

        print("Generate eposide data for training.")
        generate_episodes(train_dict_x, train_dict_y, "training", self.support_size, self.query_size, self.max_len)
        print("Generate eposide data for validation.")
        generate_episodes(valid_dict_x, valid_dict_y, "validation", self.support_size, self.query_size, self.max_len)
        print("Generate eposide data for testing.")
        generate_episodes(test_dict_x, test_dict_y, "testing", self.support_size, self.query_size, self.max_len)
        return len(drugids), len(targetids)
