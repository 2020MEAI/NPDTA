import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


class Target(torch.nn.Module):
    def __init__(self, config):
        super(Target, self).__init__()
        self.feature_dim = config['tf_dim']
        self.first_embedding_dim = config['first_embedding_dim']
        self.second_embedding_dim = config['second_embedding_dim']

        self.first_embedding_layer = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.first_embedding_dim,
            bias=True
        )

        self.second_embedding_layer = torch.nn.Linear(
            in_features=self.first_embedding_dim,
            out_features=self.second_embedding_dim,
            bias=True
        )

    def forward(self, x):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)


class Drug(torch.nn.Module):
    def __init__(self, config):
        super(Drug, self).__init__()
        self.feature_dim = config['df_dim']
        self.first_embedding_dim = config['first_embedding_dim']
        self.second_embedding_dim = config['second_embedding_dim']

        self.first_embedding_layer = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.first_embedding_dim,
            bias=True
        )

        self.second_embedding_layer = torch.nn.Linear(
            in_features=self.first_embedding_dim,
            out_features=self.second_embedding_dim,
            bias=True
        )

    def forward(self, x):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)


class AuxTarget(nn.Module):
    def __init__(self, config):
        super(AuxTarget, self).__init__()
        self.target_dim = config['tf_dim']
        self.top_m = config['top_m']
        self.attn_dim = config['second_embedding_dim']

        self.linear_query = nn.Linear(self.target_dim, self.attn_dim)
        self.linear_key = nn.Linear(self.target_dim, self.attn_dim)
        self.linear_value = nn.Linear(self.target_dim, self.attn_dim)
        self.scale = math.sqrt(self.attn_dim)

    def forward(self, query, aux_concat):
        L = query.size(0)
        keys = aux_concat.view(L, self.top_m, self.target_dim)

        Q = self.linear_query(query)  # [L, attn_dim]
        K = self.linear_key(keys)  # [L, n, attn_dim]
        V = self.linear_value(keys)  # [L, n, attn_dim]

        Q = Q.unsqueeze(1)  # [L, 1, attn_dim]
        scores = torch.matmul(Q, K.transpose(1, 2)).squeeze(1)  # [L, n]
        scores = scores / self.scale

        attn_weights = F.softmax(scores, dim=-1)  # [L, n]

        attn_weights = attn_weights.unsqueeze(1)  # [L, 1, n]
        aux_embedding = torch.matmul(attn_weights, V).squeeze(1)  # [L, attn_dim]
        return aux_embedding


class Expert(nn.Module):
    # Maps an (x_i, y_i) pair to a representation r_i.
    def __init__(self, x_dim, y_dim, h1_dim, h2_dim, z1_dim, dropout_rate):
        super(Expert, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.z1_dim = z1_dim
        self.dropout_rate = dropout_rate

        layers = [nn.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h1_dim, self.h2_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h2_dim, self.z1_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, input_pairs):
        return self.input_to_hidden(input_pairs)


class MoE(nn.Module):
    def __init__(self, expert_cls, x_dim, y_dim, h1_dim, h2_dim, z1_dim, dropout_rate, top_e=3, num_e=6):
        super(MoE, self).__init__()

        self.top_e = top_e
        self.num_e = num_e

        # Instantiate experts
        self.experts = nn.ModuleList([
            expert_cls(x_dim, y_dim, h1_dim, h2_dim, z1_dim, dropout_rate) for _ in range(self.num_e)
        ])

        # Gate network to produce routing scores
        self.gate = nn.Linear(x_dim + y_dim, self.num_e)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        y = y.view(-1, 1)
        input_pairs = torch.cat((x, y), dim=1)
        batch_size = input_pairs.size(0)

        # Compute gate logits & probabilities
        gate_logits = self.gate(input_pairs)  # (batch_size, num_experts)
        gate_probs = self.softmax(gate_logits)  # (batch_size, num_experts)
        # Select top-e experts per sample
        tope_probs, tope_idx = torch.topk(gate_probs, self.top_e, dim=-1)  # (batch_size, top_e)

        # Collect expert outputs
        rs = []
        for expert in self.experts:
            r = expert(input_pairs)  # (batch_size, z1_dim)
            rs.append(r)
        # Stack along expert dimension
        rs = torch.stack(rs, dim=1)  # (batch_size, num_experts, z1_dim)

        # Prepare batch indices for gathering
        batch_idx = torch.arange(batch_size, device=input_pairs.device).unsqueeze(-1)  # (batch_size, 1)
        # Gather top-k expert outputs
        selected_rs = rs[batch_idx, tope_idx]  # (batch_size, top_e, z1_dim)

        # Normalize top-e probabilities
        norm_probs = tope_probs / tope_probs.sum(dim=-1, keepdim=True)  # (batch_size, top_e)
        norm_probs = norm_probs.unsqueeze(-1)  # (batch_size, top_e, 1)

        # Weighted aggregation of expert outputs
        r_out = (selected_rs * norm_probs).sum(dim=1)

        return r_out


class MuSigmaEncoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z_dim = z_dim

        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, self.z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, self.z_dim)

    def forward(self, z_input):
        hidden = torch.relu(self.z_to_hidden(z_input))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, log_sigma, z


class GetH(nn.Module):
    def __init__(self, x_dim, y_dim, h1_dim, h2_dim, final_dim, dropout_rate):
        super(GetH, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.final_dim = final_dim
        self.dropout_rate = dropout_rate
        layers = [nn.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h1_dim, self.h2_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h2_dim, self.final_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        y = y.view(-1, 1)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class PPB(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature):
        super(PPB, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.bank = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))

    def forward(self, h_embed):
        res = torch.norm(h_embed-self.bank, p=2, dim=1, keepdim=True)
        res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # 1*k
        U = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(U, self.bank)
        # simple add operation
        g_embed = value + h_embed
        # calculate target distribution
        return U, g_embed


class AP(nn.Module):
    """
    Maps target input x_target and z, r to predictions y_target.
    """
    def __init__(self, x_dim, z_dim, g_dim, h1_dim, h2_dim, h3_dim, y_dim, dropout_rate):
        super(AP, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.y_dim = y_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden_layer_1 = nn.Linear(self.x_dim + self.z_dim, self.h1_dim)
        self.hidden_layer_2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.hidden_layer_3 = nn.Linear(self.h2_dim, self.h3_dim)

        self.film_layer_1_beta = nn.Linear(self.g_dim, self.h1_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.g_dim, self.h1_dim, bias=False)
        self.film_layer_2_beta = nn.Linear(self.g_dim, self.h2_dim, bias=False)
        self.film_layer_2_gamma = nn.Linear(self.g_dim, self.h2_dim, bias=False)
        self.film_layer_3_beta = nn.Linear(self.g_dim, self.h3_dim, bias=False)
        self.film_layer_3_gamma = nn.Linear(self.g_dim, self.h3_dim, bias=False)

        self.final_projection = nn.Linear(self.h3_dim, self.y_dim)

    def forward(self, x, z, task):
        interaction_size, _ = x.size()
        z = z.unsqueeze(0).repeat(interaction_size, 1)
        # Input is concatenation of z with every row of x
        inputs = torch.cat((x, z), dim=1)
        hidden_1 = self.hidden_layer_1(inputs)
        beta_1 = torch.tanh(self.film_layer_1_beta(task))
        gamma_1 = torch.tanh(self.film_layer_1_gamma(task))
        hidden_1 = torch.mul(hidden_1, gamma_1) + beta_1
        hidden_1 = self.dropout(hidden_1)
        hidden_2 = F.relu(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_2)
        beta_2 = torch.tanh(self.film_layer_2_beta(task))
        gamma_2 = torch.tanh(self.film_layer_2_gamma(task))
        hidden_2 = torch.mul(hidden_2, gamma_2) + beta_2
        hidden_2 = self.dropout(hidden_2)
        hidden_3 = F.relu(hidden_2)

        hidden_3 = self.hidden_layer_3(hidden_3)
        beta_3 = torch.tanh(self.film_layer_3_beta(task))
        gamma_3 = torch.tanh(self.film_layer_3_gamma(task))
        hidden_final = torch.mul(hidden_3, gamma_3) + beta_3
        hidden_final = self.dropout(hidden_final)
        hidden_final = F.relu(hidden_final)

        y_pred = self.final_projection(hidden_final)
        return y_pred
