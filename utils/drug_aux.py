import os
import pickle
import torch
import torch.nn.functional as F


def read_support_set(support_dir):
    set_size = int(len(os.listdir(support_dir)) / 4)
    supp_xs_s = []
    for idx in range(set_size):
        supp_xs_s.append(pickle.load(open("{}/supp_x_{}.pkl".format(support_dir, idx), "rb")))
    support_set = torch.cat(supp_xs_s, dim=0)
    return support_set


def compute_and_store_drug_similarity(support_set, output_file):
    drug_emb = support_set[:, 320:]  # shape: [M, 384]
    drug_emb = torch.unique(drug_emb, dim=0)

    drug_emb_norm = F.normalize(drug_emb, p=2, dim=1)
    similarity_matrix = torch.mm(drug_emb_norm, drug_emb_norm.t())

    similarity_dict = {
        "drug_embeddings": drug_emb.cpu(),
        "similarity_matrix": similarity_matrix.cpu()
    }

    with open(output_file, 'wb') as f:
        pickle.dump(similarity_dict, f)

    print(f"Stored {drug_emb.size(0)} unique drug embeddings and similarity matrix to {output_file}")


def filter_drugs_by_similarity(similarity_matrix, target_index, threshold=0.8):
    sim_row = similarity_matrix[target_index]  # shape: [M]
    indices = (sim_row >= threshold).nonzero(as_tuple=True)[0]
    indices = indices[indices != target_index]
    return indices


def get_aux_drug(dataset_path, category):
    support_dir = dataset_path + "/" + category + "/log"
    output_file = dataset_path + "/drug_similarity_" + category + ".pkl"
    support_set = read_support_set(support_dir)
    compute_and_store_drug_similarity(support_set, output_file)
