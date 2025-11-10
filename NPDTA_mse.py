import torch
import numpy as np
from random import randint
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from embeddings_NPDTA import Target, Drug, AuxTarget, Expert, MoE, MuSigmaEncoder, GetH, PPB, AP
import torch.nn as nn
import pickle


class NP(nn.Module):
    def __init__(self, config):
        super(NP, self).__init__()
        self.x_dim = config['second_embedding_dim'] * 4
        # use one-hot or not?
        self.y_dim = 1
        self.z1_dim = config['z1_dim']
        self.z2_dim = config['z2_dim']
        # z is the dimension size of mu and sigma.
        self.z_dim = config['z_dim']
        # the dimension size of rc.
        self.e_h1_dim = config['e_h1_dim']
        self.e_h2_dim = config['e_h2_dim']
        self.top_e = config['top_e']
        self.num_e = config['num_e']

        self.h_h1_dim = config['h_h1_dim']
        self.h_h2_dim = config['h_h2_dim']
        self.h_final_dim = config['h_final_dim']

        self.l_h1_dim = config['l_h1_dim']
        self.l_h2_dim = config['l_h2_dim']
        self.l_h3_dim = config['l_h3_dim']

        self.clusters_k = config['clusters_k']
        self.temperture = config['temperature']
        self.dropout_rate = config['dropout_rate']

        # Initialize networks
        self.target_emb = Target(config)
        self.drug_emb = Drug(config)
        self.aux_target_emb = AuxTarget(config)
        # This encoder is used to generated z actually, it is a latent encoder in ANP.
        self.xy_to_r = MoE(Expert, self.x_dim, self.y_dim, self.e_h1_dim, self.e_h2_dim, self.z1_dim, self.dropout_rate, self.top_e, self.num_e)
        self.z_to_mu_sigma = MuSigmaEncoder(self.z1_dim, self.z2_dim, self.z_dim)
        # This encoder is used to generated r actually, it is a deterministic encoder in ANP.
        self.xy_to_h = GetH(self.x_dim, self.y_dim, self.h_h1_dim, self.h_h2_dim, self.h_final_dim, self.dropout_rate)
        self.ppb = PPB(self.clusters_k, self.h_final_dim, self.temperture)
        self.xz_to_y = AP(self.x_dim, self.z_dim, self.h_final_dim, self.l_h1_dim, self.l_h2_dim, self.l_h3_dim, self.y_dim, self.dropout_rate)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=0)

    def xy_to_mu_sigma(self, x, y):
        # Encode each point into a representation r_i
        r_i = self.xy_to_r(x, y)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.z_to_mu_sigma(r)

    # embedding each (target, drug) as the x for np
    def embedding(self, x, aux_drug, aux_target):
        tf_dim = self.target_emb.feature_dim
        target_x = Variable(x[:, 0:tf_dim], requires_grad=False).float()
        drug_x = Variable(x[:, tf_dim:], requires_grad=False).float()
        aux_drug = aux_drug.repeat(drug_x.shape[0], 1)
        target_emb = self.target_emb(target_x)
        drug_emb = self.drug_emb(drug_x)
        aux_drug_emb = self.drug_emb(aux_drug)
        aux_target_emb = self.aux_target_emb(target_x, aux_target)
        x = torch.cat((aux_target_emb, target_emb, drug_emb, aux_drug_emb), 1)
        return x

    def forward(self, x_context, y_context, x_target, y_target, aux_drug, aux_target_s, aux_target_q):
        x_context_embed = self.embedding(x_context, aux_drug, aux_target_s)
        x_target_embed = self.embedding(x_target, aux_drug, aux_target_q)

        if self.training:
            # sigma is log_sigma actually
            mu_target, sigma_target, z_target = self.xy_to_mu_sigma(x_target_embed, y_target)
            # mu_context, sigma_context, z_context = self.xy_aux_to_mu_sigma(x_context_embed, y_context, x_aux_embed, y_aux)
            mu_context, sigma_context, z_context = self.xy_to_mu_sigma(x_context_embed, y_context)
            h_i = self.xy_to_h(x_context_embed, y_context)
            h = self.aggregate(h_i)
            U_distribution, g = self.ppb(h)
            p_y_pred = self.xz_to_y(x_target_embed, z_target, g)
            return p_y_pred, mu_target, sigma_target, mu_context, sigma_context, U_distribution
        else:
            # mu_context, sigma_context, z_context = self.xy_aux_to_mu_sigma(x_context_embed, y_context, x_aux_embed, y_aux)
            mu_context, sigma_context, z_context = self.xy_to_mu_sigma(x_context_embed, y_context)
            h_i = self.xy_to_h(x_context_embed, y_context)
            h = self.aggregate(h_i)
            U_distribution, g = self.ppb(h)
            p_y_pred = self.xz_to_y(x_target_embed, z_context, g)
            return p_y_pred


class Trainer(torch.nn.Module):
    def __init__(self, config):
        self.opt = config
        super(Trainer, self).__init__()
        self.use_cuda = config['use_cuda']
        self.np = NP(self.opt)
        self._lambda = config['lambda']
        self.optimizer = torch.optim.Adam(self.np.parameters(), lr=config['lr'])
        self.target_dim = config['tf_dim']
        self.drug_dim = config['df_dim']
        self.data_dir = config['data_dir']
        self.top_n = config['top_n']
        self.top_m = config['top_m']

    def new_kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

    def loss(self, p_y_pred, y_target, mu_target, sigma_target, mu_context, sigma_context):
        # regression_loss = F.mse_loss(p_y_pred, y_target.view(-1, 1))
        weights = torch.exp(0.1 * (y_target.view(-1, 1) - 6))
        weighted_mse = weights * (p_y_pred - y_target.view(-1, 1)) ** 2
        regression_loss = torch.mean(weighted_mse)
        # kl divergence between target and context
        kl = self.new_kl_div(mu_context, sigma_context, mu_target, sigma_target)
        return regression_loss+kl

    def new_context_target_split(self, support_set_x, support_set_y, query_set_x, query_set_y):
        total_x = torch.cat((support_set_x, query_set_x), 0)
        total_y = torch.cat((support_set_y, query_set_y), 0)
        total_size = total_x.size(0)
        context_min = self.opt['context_min']
        num_context = np.random.randint(context_min, total_size)
        num_target = np.random.randint(0, total_size - num_context)
        sampled = np.random.choice(total_size, num_context+num_target, replace=False)
        x_context = total_x[sampled[:num_context], :]
        y_context = total_y[sampled[:num_context]]
        x_target = total_x[sampled, :]
        y_target = total_y[sampled]
        return x_context, y_context, x_target, y_target

    def filter_auxiliary_pairs(self, target_support_x, all_train_support_x, all_train_support_y, threshold=0.95):
        device = target_support_x.device
        target_drug = target_support_x[0, self.target_dim:].unsqueeze(0)

        aux_x_list = []
        aux_y_list = []

        for task_x, task_y in zip(all_train_support_x, all_train_support_y):
            candidate_drug = task_x[0, self.target_dim:].unsqueeze(0)
            if torch.equal(target_drug, candidate_drug):
                continue
            sim = F.cosine_similarity(target_drug, candidate_drug, dim=1)
            if sim.item() >= threshold:
                aux_x_list.append(task_x)
                aux_y_list.append(task_y)

        if len(aux_x_list) > 0:
            aux_x = torch.cat(aux_x_list, dim=0)
            aux_y = torch.cat(aux_y_list, dim=0)
        else:
            aux_x = torch.empty(0, target_support_x.size(1), device=device)
            aux_y = torch.empty(0, device=device)

        return aux_x, aux_y

    def query_similar_drugs(self, query_drug, similarity_dict, top_n=3):
        drug_embeddings = similarity_dict["drug_embeddings"]
        similarity_matrix = similarity_dict["similarity_matrix"]
        if self.use_cuda:
            drug_embeddings = drug_embeddings.cuda()
            similarity_matrix = similarity_matrix.cuda()

        found_index = None
        for i in range(drug_embeddings.size(0)):
            if torch.equal(drug_embeddings[i], query_drug):
                found_index = i
                break
        if found_index is None:
            raise ValueError("The query drug was not found in the stored drug embeddings.")

        sim_row = similarity_matrix[found_index].clone()  # shape: [M]
        sim_row[found_index] = 0.0

        topk_values, topk_indices = torch.topk(sim_row, top_n)
        similar_drug_embs = drug_embeddings[topk_indices]
        similar_drug_emb = similar_drug_embs.mean(dim=0, keepdim=True)

        return similar_drug_emb

    def assemble_auxiliary_targets(self, target_embeddings, top_m=5):
        L, d = target_embeddings.size()
        device = target_embeddings.device
        aux_list = []
        for i in range(L):
            if i == 0:
                aux = torch.zeros(top_m * d, device=device)
            else:
                start = max(0, i - top_m)
                aux_set = target_embeddings[start:i]  # shape: [m, d]，m <= top_m
                m = aux_set.size(0)
                if m < top_m:
                    pad = torch.zeros(top_m - m, d, device=device)
                    aux_set = torch.cat([aux_set, pad], dim=0)
                aux = aux_set.reshape(-1)
            aux_list.append(aux.unsqueeze(0))
        aux_targets = torch.cat(aux_list, dim=0)
        return aux_targets

    def assemble_auxiliary_targets_test(self, query_targets, support_targets, top_m=5):
        L_support, d = support_targets.size()
        device = support_targets.device
        if L_support < top_m:
            pad = torch.zeros(top_m - L_support, d, device=device)
            aux_set = torch.cat([support_targets, pad], dim=0)
        else:
            aux_set = support_targets[-top_m:]
        aux_concat = aux_set.reshape(-1)
        L_query = query_targets.size(0)
        aux_targets = aux_concat.unsqueeze(0).expand(L_query, -1)
        return aux_targets

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, train_dict):
        batch_sz = len(support_set_xs)
        losses = []
        U_distribs = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            # aux_x, aux_y = self.filter_auxiliary_pairs(support_set_xs[i], support_set_xs, support_set_ys)
            aux_drug = self.query_similar_drugs(support_set_xs[i][0, self.target_dim:], train_dict, self.top_n)
            x_context, y_context, x_target, y_target = self.new_context_target_split(support_set_xs[i], support_set_ys[i],
                                                                                     query_set_xs[i], query_set_ys[i])
            aux_target_s = self.assemble_auxiliary_targets(x_context[:, 0:self.target_dim], self.top_m)
            aux_target_q = self.assemble_auxiliary_targets(x_target[:, 0:self.target_dim], self.top_m)
            p_y_pred, mu_target, sigma_target, mu_context, sigma_context, U_distribution = self.np(x_context, y_context, x_target, y_target,
                                                                                                   aux_drug, aux_target_s, aux_target_q)
            U_distribs.append(U_distribution)
            loss = self.loss(p_y_pred, y_target, mu_target, sigma_target, mu_context, sigma_context)
            #print('Each task has loss: ', loss)
            losses.append(loss)
        # calculate target distribution for clustering in batch manner.
        U_distribs = torch.stack(U_distribs)  # batchsize * k
        U_distribs_sq = torch.pow(U_distribs, 2)  # batchsize * k
        U_distribs_sum = torch.sum(U_distribs, dim=0, keepdim=True)  # 1*k
        temp = U_distribs_sq / U_distribs_sum  # batchsize * k
        temp_sum = torch.sum(temp, dim=1, keepdim=True)  # batchsize * 1
        target_distribs = temp / temp_sum
        # calculate the kl loss
        clustering_loss = self._lambda * F.kl_div(U_distribs.log(), target_distribs, reduction='batchmean')
        #print('The clustering loss is %.6f' % (clustering_loss.item()))
        np_losses_mean = torch.stack(losses).mean(0)
        total_loss = np_losses_mean + clustering_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def query_rec(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, test_dict):
        batch_sz = 1
        # used for calculating the rmse.
        losses_q = []
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            # aux_x, aux_y = self.filter_auxiliary_pairs(support_set_xs[i], support_set_xs, support_set_ys)
            aux_drug = self.query_similar_drugs(support_set_xs[i][0, self.target_dim:], test_dict, self.top_n)
            aux_target_s = self.assemble_auxiliary_targets(support_set_xs[i][:, 0:self.target_dim], self.top_m)
            aux_target_q = self.assemble_auxiliary_targets_test(query_set_xs[i][:, 0:self.target_dim], support_set_xs[i][:, 0:self.target_dim], self.top_m)
            query_set_y_pred = self.np(support_set_xs[i], support_set_ys[i], query_set_xs[i], query_set_ys[i],
                                       aux_drug, aux_target_s, aux_target_q)
            # loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            weights = torch.exp(0.1 * (query_set_ys[i].view(-1, 1) - 6))
            weighted_mse = weights * (query_set_y_pred - query_set_ys[i].view(-1, 1)) ** 2
            loss_q = torch.mean(weighted_mse)
            losses_q.append(loss_q)
            total_preds = torch.cat((total_preds, query_set_y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, query_set_ys[i].view(-1, 1).cpu()), 0)
        losses_q = torch.stack(losses_q).mean(0)
        return losses_q.item(), total_preds.detach().numpy().flatten(), total_labels.detach().numpy().flatten()
