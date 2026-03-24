from easydl import aToBSheduler
import scipy.sparse as sp
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import random
from torch.backends import cudnn
import numpy as np
import torch.nn.modules.loss
from tqdm import tqdm
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import ot
import os
import warnings

warnings.filterwarnings('ignore')


def mclust_R(adata, n_clusters, use_rep='DuaST', key_added='DuaST', random_seed=2023):
    '''
    Perform clustering using the mclust in R.
    '''
    modelNames = 'EEE'

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def generate_adj_mat(adata, include_self=False, n=6):
    '''
    Construct spatial adjacency matrix using KNN based on spatial coordinates.
    Each spot is connected to its n nearest neighbors.
    '''

    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n + 1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
    '''
    Normalize adjacency matrix using symmetric normalization:
    '''
    adj_ = adj + sp.eye(adj.shape[0])
    adj_ori_numpy = adj_.toarray()
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized), adj_ori_numpy


def graph_construction(adata, n=6):
    '''
    Build graph structure used in DuaST.
    '''
    adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    adj_m1 = sp.coo_matrix(adj_m1)

    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    adj_norm_m1, adj_ori_numpy = preprocess_graph(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])

    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1.coalesce(),
        "norm_value": norm_m1,
        "adj_ori_numpy": adj_ori_numpy
    }

    return graph_dict


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class GradientReverseLayer(torch.autograd.Function):
    '''
    Gradient Reversal Layer (GRL).
    '''

    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)


class AdvNet(nn.Module):
    '''
    Adversarial discriminator
    '''

    def __init__(self, in_feature=128, hidden_size=64):
        super(AdvNet, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0,
                                                                   gamma=10,
                                                                   max_iter=self.max_iter))

    def forward(self, x, reverse=True):
        if reverse:
            x = self.grl(x)
        x = self.ad_layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class GraphConvolution(Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class Discriminator_dgi(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_dgi, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class Attention(nn.Module):
    '''
    Attention-based fusion module.
    '''

    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class DuaST_module(nn.Module):
    '''
    Dual-branch framework for spatial transcriptomics.

    Components:
    - Spatial-aware branch (GCN-based)
    - Non-spatial branch (MLP-based)
    - Contrastive learning (local & global)
    - Adversarial alignment (AAM)
    - Attention fusion (AFM)
    '''

    def __init__(
            self,
            input_dim,
            graph_neigh,
            feat_hidden0=512,
            feat_hidden1=64,
            feat_hidden2=32,
            gcn_hidden0=512,
            gcn_hidden1=64,
            gcn_hidden2=32,
            p_drop=0,
    ):
        super(DuaST_module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden0 = feat_hidden0
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden0 = gcn_hidden0
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.latent_dim = self.gcn_hidden2

        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.feat_hidden0),
            torch.nn.BatchNorm1d(self.feat_hidden0),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feat_hidden0, self.feat_hidden2),
            torch.nn.BatchNorm1d(self.feat_hidden2),
            torch.nn.ReLU())

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.feat_hidden2, self.feat_hidden0),
            torch.nn.BatchNorm1d(self.feat_hidden0),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feat_hidden0, self.input_dim))

        self.gc1 = GraphConvolution(self.input_dim, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder()

        self.graph_neigh = graph_neigh
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.disc = Discriminator_dgi(self.gcn_hidden2)
        self.advnet = AdvNet(in_feature=self.gcn_hidden2, hidden_size=16)
        self.attention = Attention(self.gcn_hidden2)

    def encode(self, x, adj):
        '''
        Encode input features into:
        - GNN latent space (spatial-aware)
        - MLP latent space (non-spatial)
        '''
        feat_liner = self.encoder_linear(x)   # non-spatial
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_liner

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        '''
        Forward pass of DuaST.
        Steps:
        1. Encode features via dual branches
        2. Sample latent representation (VGAE)
        3. Contrastive learning (local + global)
        4. Attention-based fusion
        5. Reconstruction (decoder)
        6. Adversarial alignment
        '''
        mu, logvar, feat_liner = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)

        perm = torch.randperm(feat_liner.size(0))
        z_n = feat_liner[perm]

        z_p = self.read(gnn_z, self.graph_neigh)
        z_p = self.sigm(z_p)
        ret1 = self.disc(z_p, gnn_z, z_n)

        z_p_2 = self.sigm(F.normalize(feat_liner, p=2, dim=1))
        ret2 = self.disc(z_p_2, gnn_z, z_n)

        z_stack = torch.stack([gnn_z, feat_liner], dim=1)
        z, _ = self.attention(z_stack)

        de_feat = self.decoder(z)

        prob_source = self.advnet.forward(gnn_z)
        prob_target = self.advnet.forward(feat_liner)
        return z, mu, logvar, de_feat, prob_source, prob_target, feat_liner, gnn_z, ret1, ret2


def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    '''
    Graph reconstruction loss (VGAE ELBO):
    '''
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    elbo = cost + KLD
    return elbo


def add_contrastive_label(n_spot):
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    return label_CSL


class DuaST:
    '''
    Training wrapper for DuaST.
    '''

    def __init__(
            self,
            X,
            graph_dict,
            graph_neigh,
            rec_w=10,
            gcn_w=0.1,
            con_w=1,
            adv_w=1,
            epochs=200,
            device='cuda:0',
    ):
        print(rec_w, gcn_w, con_w, adv_w, epochs)
        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.con_w = con_w
        self.adv_w = adv_w
        self.epochs = epochs
        self.device = device
        self.graph_neigh = graph_neigh.to(self.device)
        self.n_spot = len(X)
        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]

        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.adj_ori_numpy = torch.FloatTensor(graph_dict["adj_ori_numpy"]).to(self.device)
        self.norm_value = graph_dict["norm_value"]

        self.model = DuaST_module(self.input_dim, self.graph_neigh).to(self.device)
        label_CSL = add_contrastive_label(self.n_spot)
        self.label_CSL = torch.FloatTensor(label_CSL).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()

    def train_loss(self):
        '''
        Train DuaST model.
        '''
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=0.001)
        self.model.train()

        for _ in tqdm(range(self.epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, prob_source, prob_target, feat_liner, gnn_z, ret1, ret2 = self.model(self.X,
                                                                                                                self.adj_norm)
            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z),
                labels=self.adj_ori_numpy,
                mu=mu,
                logvar=logvar,
                n_nodes=self.n_spot,
                norm=self.norm_value,
            )

            loss_sl_att_local = self.loss_CSL(ret1, self.label_CSL)
            loss_sl_att_global = self.loss_CSL(ret2, self.label_CSL)
            loss_rec = F.mse_loss(de_feat, self.X)

            wasserstein_distance = (prob_source.mean() - prob_target.mean())
            adv_loss = -wasserstein_distance
            loss = self.rec_w * loss_rec + self.adv_w * adv_loss + self.con_w * (
                    loss_sl_att_local + loss_sl_att_global) + self.gcn_w * loss_gcn
            loss.backward()
            self.optimizer.step()

    def process(self):
        '''
        Standard preprocessing.
        '''
        self.model.eval()
        latent_z, mu, logvar, de_feat, prob_source, prob_target, feat_liner, gnn_z, ret1, ret2 = self.model(self.X,
                                                                                                            self.adj_norm)

        latent_z = latent_z.data.cpu().numpy()

        return latent_z, de_feat


def construct_interaction(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction


def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.log1p(adata)
    sc.pp.scale(adata)


def refine_label(adata, radius=50, key='label'):
    '''
    Spatial smoothing of cluster labels.
    Only for HBC dataset.
    '''
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'HBC'
    os.environ['R_HOME'] = '/home/liangxiao/miniconda3/envs/R/lib/R'
    datatype = '10X'
    file_fold = '/home/liangxiao/lllxxx/Human_Breast_Cancer/'
    adata = sc.read_visium(file_fold)
    adata.var_names_make_unique()
    adata_2 = adata.copy()
    print(adata)
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata.obs['ground_truth'] = df_meta_layer.values
    n_clusters = len(adata.obs['ground_truth'].unique())
    # print('n_clusters:', n_clusters)

    preprocess(adata)
    adata = adata[:, adata.var['highly_variable']]
    feat_liner = adata.X.toarray()
    adata.obsm['feat_liner'] = feat_liner

    if 'graph_neigh' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata)
        else:
            construct_interaction(adata)

    graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(feat_liner.shape[0]))
    graph_dict = graph_construction(adata, 6)

    res_all = []
    l1 = 20
    l2 = 5
    l3 = 1
    l4 = 0.01

    fix_seed(2025)

    DuaST_net = DuaST(adata.obsm['feat_liner'], graph_dict, graph_neigh, rec_w=l1, gcn_w=l2, con_w=l3,
                      adv_w=l4, epochs=300, device=device)

    DuaST_net.train_loss()
    DuaST_feat, de_feat = DuaST_net.process()
    adata.obsm['DuaST'] = DuaST_feat

    mclust_R(adata, n_clusters, use_rep='DuaST', key_added='DuaST')

    # only for HBC
    adata.obs['domain'] = adata.obs['DuaST'].copy()
    adata.obs['domain'] = adata.obs['domain'].astype('category')

    if dataset == 'HBC':
        new_type = refine_label(adata, radius=50, key='domain')
    else:
        new_type = adata.obs['domain']
    ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], new_type)
    NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], new_type)
    print('ARI', ARI)
    print('NMI', NMI)
