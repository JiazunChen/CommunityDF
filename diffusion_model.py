import torch
import torch.nn as nn
import torch.nn.functional as F
from noise_schedule import PredefinedNoiseSchedule
from gat import Swish, Sigmoid, GAT, GraphTransformer, PredictComsize, make_linear_block
from utils import PlaceHolder
from torch_geometric.utils import to_dense_adj, to_dense_batch
import diffusion_utils


def my_one_hot(x, class_num):
    if class_num == 1:
        return x.reshape(-1, 1)
    else:
        return F.one_hot(x, class_num)


def to_dense(batch, x):
    x, node_mask = to_dense_batch(x, batch['graph'].ndata['index'])
    return PlaceHolder(X=x), node_mask


class DenoisingDiffusion(nn.Module):
    def __init__(self, args, input, output, heads, device, dropout=0.0, diffusion_noise_schedule='cosine',
                 diffusion_steps=500, margin=[0.5, 0.5]):
        super().__init__()
        self.args = args
        self.gamma = PredefinedNoiseSchedule(diffusion_noise_schedule, timesteps=args.diffusion_steps)
        self.T = diffusion_steps
        self.device = device
        if args.feat_dim != 0:
            self.feat_embedding = nn.Linear(args.feat_dim, input, bias=False)
        self.seed_embedding = nn.Linear(1, input, bias=False)
        self.node_embedding = nn.Linear(1, input, bias=False)
        self.time_embedding = nn.Linear(1, input, bias=False)
        self.degree_embedding = nn.Linear(1, input, bias=False)
        self.cluster_embedding = nn.Linear(1, input, bias=False)
        if self.args.competitor is not None:
            self.pg_embedding = nn.Linear(1, input, bias=False)
        if args.pos_enc_dim is not None:
            self.pos_embedding = nn.Linear(args.pos_enc_dim, input, bias=False)
        self.gat = GraphTransformer(input, output, heads, dropout)

        mlpinput = output
        self.mlp = nn.Sequential(
            make_linear_block(mlpinput, output, nn.ReLU, None, dropout=dropout),
            make_linear_block(output, 1, None, None, dropout=dropout),
        )

    def get_emd(self, graph, X, Seed, T, D, C):
        graph = graph.to(self.device)
        X = self.node_embedding(X.to(self.device))
        Seed = self.seed_embedding(Seed.to(self.device))
        T = self.time_embedding(T.to(self.device))
        D = self.degree_embedding(D.to(self.device))
        C = self.cluster_embedding(C.to(self.device))
        input_X = X + Seed + T + D + C
        if self.args.competitor is not None:
            pg_embedding = self.pg_embedding(graph.ndata['pg'].reshape(-1, 1))
            input_X += pg_embedding

        if self.args.pos_enc_dim:
            Pos = self.pos_embedding(graph.ndata['lap_pe'])
            input_X += Pos

        if self.args.feat_dim != 0:
            nodefeat = graph.ndata['feat'].float()
            nodefeat = self.feat_embedding(nodefeat.to(self.device))
            input_X += nodefeat

        nodeemd = self.gat(graph, input_X)
        # print(nodeemd.shape)

        return nodeemd

    def forward(self, graph, X, Seed, T, D, C):
        graph = graph.to(self.device)
        X = self.node_embedding(X.to(self.device))
        Seed = self.seed_embedding(Seed.to(self.device))
        T = self.time_embedding(T.to(self.device))
        D = self.degree_embedding(D.to(self.device))
        C = self.cluster_embedding(C.to(self.device))
        input_X = X + Seed + T + D + C
        if self.args.competitor is not None:
            pg_embedding = self.pg_embedding(graph.ndata['pg'].reshape(-1, 1))
            input_X += pg_embedding

        if self.args.pos_enc_dim:
            Pos = self.pos_embedding(graph.ndata['lap_pe'])
            input_X += Pos

        if self.args.feat_dim != 0:
            nodefeat = graph.ndata['feat'].float()
            nodefeat = self.feat_embedding(nodefeat.to(self.device))
            input_X += nodefeat

        nodeemd = self.gat(graph, input_X)
        newnodeemd = nodeemd
        predX = self.mlp(newnodeemd)

        return predX

    def apply_noise(self, X, node_mask, pg=None):
        """ Sample noise and apply it to the data. """
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1

        # Sample a timestep t.
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1))

        t_int = t_int.type_as(X).float()  # (bs, 1)
        s_int = t_int - 1

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s_normalized = s_int / self.T
        t_normalized = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = diffusion_utils.inflate_batch_array(self.gamma(s_normalized.to(self.device)), X.size())  # (bs, 1, 1),
        gamma_t = diffusion_utils.inflate_batch_array(self.gamma(t_normalized.to(self.device)), X.size())  # (bs, 1, 1)

        # Compute alpha_t and sigma_t from gamma, with correct size for X, E and z
        alpha_t = diffusion_utils.alpha(gamma_t, X.size())  # (bs, 1, ..., 1), same n_dims than X
        sigma_t = diffusion_utils.sigma(gamma_t, X.size())  # (bs, 1, ..., 1), same n_dims than X

        # Sample zt ~ Normal(alpha_t x, sigma_t)

        if self.args.competitor == 'grad':
            eps = diffusion_utils.sample_feature_noise(X.size(), node_mask).type_as(X)
            # Sample z_t given x, h for timestep t, from q(z_t | x, h)
            X_t = alpha_t * (X - pg.to(X.device)) + sigma_t * eps.X
        else:
            eps = diffusion_utils.sample_feature_noise(X.size(), node_mask, pg).type_as(X)
            # Sample z_t given x, h for timestep t, from q(z_t | x, h)
            X_t = alpha_t * X + sigma_t * eps.X

        noisy_data = {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
                      'epsX': eps.X,
                      'X_t': X_t, 't_int': t_int}
        return noisy_data

    def sample_p_zs_given_zt(self, s, t, X_t, node_mask, batch, pg):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        X_t = X_t.to(self.device)
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(gamma_t,
                                                                                                       gamma_s,
                                                                                                       X_t.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=X_t.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=X_t.size())

        noisy_data = {'X_t': X_t, 't': t}
        inputT = noisy_data['t'].repeat(1, noisy_data['X_t'].shape[1])[node_mask.cpu()].float().view(-1, 1)
        inputX = noisy_data['X_t'][node_mask].float().view(-1, 1)
        inputS = batch['graph'].ndata['seed'].float().view(-1, 1)

        inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
        inputD = (inputD - self.args.mean_triangles) / self.args.std_triangles

        inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)

        epsX = self.forward(batch['graph'], inputX, inputS, inputT, inputD, inputC)

        eps, new_node_mask = to_dense(batch, epsX)

        assert new_node_mask.equal(node_mask)
        # Compute mu for p(zs | zt).

        # print(X_t.device,alpha_t_given_s.device,sigma_t.device,eps.X.device)
        mu_X = X_t / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps.X

        if self.args.ori_xc_coefficient:
            # print(diffusion_utils.get_xc_item(gamma_t,gamma_s,X_t.size())[0].cpu())
            mu_X += diffusion_utils.get_xc_item(gamma_t, gamma_s, X_t.size()) * pg.to(mu_X.device)

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        # print(' xc',diffusion_utils.get_xc_item(gamma_t,gamma_s,X_t.size())[0],'sigma',sigma[0] )
        # Sample zs given the parameters derived from zt.
        if self.args.sample_place == 'first' or self.args.competitor == 'grad':
            z_s = diffusion_utils.sample_normal(mu_X, sigma, node_mask).type_as(mu_X)
        else:
            z_s = diffusion_utils.sample_normal(mu_X, sigma, node_mask, pg).type_as(mu_X)
        return z_s

    def sample_discrete_graph_given_z0(self, X_0, node_mask, batch, pg):
        """ Samples X, E, y ~ p(X, E, y|z0): once the diffusion is done, we need to map the result
        to categorical values.
        """
        zeros = torch.zeros(size=(X_0.size(0), 1), device=X_0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
        noisy_data = {'X_t': X_0, 't': torch.zeros(X_0.shape[0], 1).float()}
        inputT = noisy_data['t'].repeat(1, noisy_data['X_t'].shape[1])[node_mask.cpu()].float().view(-1, 1)
        inputX = noisy_data['X_t'][node_mask].float().view(-1, 1)
        inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
        inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
        inputD = (inputD - self.args.mean_triangles) / self.args.std_triangles
        inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)
        X = self.forward(batch['graph'], inputX, inputS, inputT, inputD, inputC)

        # Compute mu for p(zs | zt).
        eps0, new_node_mask = to_dense(batch, X)

        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

        pred_X = 1. / alpha_0 * (X_0 - sigma_0 * eps0.X)

        if self.args.sample_place == 'first' or self.args.competitor == 'grad':
            sampled = diffusion_utils.sample_normal(pred_X, sigma, node_mask).type_as(pred_X)
        else:
            sampled = diffusion_utils.sample_normal(pred_X, sigma, node_mask, pg).type_as(pred_X)
        if self.args.competitor == 'grad':
            sampled.X += pg.to(sampled.X.device)
        return sampled.X

    def sample_batch(self, batch, Discriminative_model=None, pre_model=None):
        batch['graph'].ndata['x'] = my_one_hot(batch['graph'].ndata['x'], 1).float()
        batch['graph'] = batch['graph'].to(self.device)
        dense_data, node_mask = to_dense(batch, batch['graph'].ndata['x'])
        X = dense_data.X.to(self.device)
        node_mask = node_mask.to(self.device)
        if self.args.guided and self.args.competitor != 'cond':
            pg, _ = to_dense(batch, batch['graph'].ndata['pg'].reshape(-1, 1))
            pg = pg.X.cpu()
        else:
            pg = None
        with torch.no_grad():
            if Discriminative_model is not None:
                Discriminative_dense_data, Discriminative_node_mask = to_dense(batch, batch['graph'].ndata['pg'])
                Discriminative_X = Discriminative_dense_data.X.to(self.device)
                Discriminative_node_mask = Discriminative_node_mask.to(self.device)
                Discriminative_inputX = Discriminative_X[Discriminative_node_mask].float().view(-1,1)
                Discriminative_inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
                Discriminative_inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
                Discriminative_inputD = (Discriminative_inputD - self.args.mean_triangles) / self.args.std_triangles
                Discriminative_inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)
                Discriminative_predX = Discriminative_model(batch['graph'], Discriminative_inputX,
                                                            Discriminative_inputS, Discriminative_inputD,
                                                            Discriminative_inputC)
                # batch['graph'].ndata['pg'] = Discriminative_predX.reshape(-1)
                pg, _ = to_dense(batch, torch.sigmoid(Discriminative_predX))
                pg = pg.X.cpu()
            if pre_model is not None:
                old_X, node_mask, _ = pre_model.sample_batch(batch, Discriminative_model)
                pg = old_X.cpu()

        z_T = diffusion_utils.sample_feature_noise(X_size=X.shape, node_mask=node_mask, pg=pg)
        X = z_T.X
        batch_size = X.shape[0]
        all_X = [X.detach().cpu()]
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).float().to(self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            z_s = self.sample_p_zs_given_zt(s=s_norm, t=t_norm, X_t=X, node_mask=node_mask, batch=batch, pg=pg)
            X = z_s.X
            all_X.append(X.detach().cpu())
            # print(X)
        finalX = self.sample_discrete_graph_given_z0(X, node_mask, batch, pg)
        # finalX = X
        all_X.append(finalX.detach().cpu())
        return finalX, node_mask, all_X
