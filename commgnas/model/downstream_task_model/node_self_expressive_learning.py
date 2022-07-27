import torch
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn import cluster
from scipy.sparse.linalg import svds
from torch.nn.parameter import Parameter
from sklearn.preprocessing import normalize

class SelfExpr(torch.nn.Module):

    def __init__(self, n):
        super(SelfExpr, self).__init__()

        self.C_ = Parameter(torch.FloatTensor(n, n).uniform_(0, 0.01))

    def forward(self, Z):

        # C = self.weight - torch.diag(torch.diagonal(self.weight))
        # because the diagonal element of C is zero
        # torch.diag(torch.diagonal(tensor_matrix_a)) : construct the diagonal matrix of tensor_matrix_a

        C = self.C_ - torch.diag(torch.diagonal(self.C_))
        output = torch.mm(C, Z)

        return C, output


class DownstreamTask(torch.nn.Module):

    def __init__(self,
                 downstream_task_parameter,
                 gnn_embedding_dim,
                 graph_data):

        super(DownstreamTask, self).__init__()

        self.data = graph_data
        self.downstream_task_parameter = downstream_task_parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semodel = SelfExpr(self.data.test_x.shape[0]).to(self.device)
        self.seoptimizer = torch.optim.Adam(self.semodel.parameters(),
                                            lr=self.downstream_task_parameter["lr"],
                                            weight_decay=self.downstream_task_parameter['weight_decay'])

    def forward(self,
                Z,
                batch_x_index,
                mode="train"):

        print("\n Starting self representation training ")

        self_representation_loss_change = self.self_representation_train(Z)

        return self_representation_loss_change

    def self_representation_train(self, Z):

        train_epoch = self.downstream_task_parameter['se_epochs']
        lambda_ = self.downstream_task_parameter['se_loss_reg']

        train_loss_value = 0.0
        first_loss = 0.0

        for epoch in range(train_epoch):

            self.semodel.train()
            self.seoptimizer.zero_grad()
            C, CZ = self.semodel(Z)

            # torch.norm calculate the 2 norm of matrix
            se_loss = torch.norm(Z - CZ)
            reg_loss = torch.norm(C)
            loss = se_loss + lambda_ * reg_loss

            loss.backward()

            self.seoptimizer.step()
            train_loss_value = loss.item()

            if epoch == 0:
                first_loss = train_loss_value
            if epoch % 10 == 0:
                print("self representation epoch: ", epoch, " train loss value: ", train_loss_value)

        final_loss = train_loss_value

        self_representation_loss_change = abs(first_loss - final_loss)

        return self_representation_loss_change

    def enhance_sim_matrix(self, C, K, d, alpha):

        # C: coefficient matrix,
        # K: number of clusters,
        # d: dimension of each subspace

        C = 0.5 * (C + C.T)
        r = min(d * K + 1, C.shape[0] - 1)
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = 0.5 * (L + L.T)
        L = L / L.max()

        return L

    def test_spectral(self, c, y_train, n_class):

        y_train_x, _ = self.post_proC(c, n_class, 4, 1)
        print("Spectral Clustering Done.. Finding Best Fit..")
        scores = self.err_rate(y_train.detach().cpu().numpy(), y_train_x)

        return scores

    def err_rate(self, gt_s, s):

        y_pred = self.best_map(gt_s, s)
        acc = metrics.accuracy_score(gt_s, y_pred)
        nmi = metrics.normalized_mutual_info_score(gt_s, y_pred)
        f1_macro = metrics.f1_score(gt_s, y_pred, average='macro')

        return [acc, nmi, f1_macro]

    def post_proC(self, C, K, d, alpha):

        # C: coefficient matrix,
        # K: number of clusters,
        # d: dimension of each subspace
        # L = self.enhance_sim_matrix(C, K, d, alpha)

        L = C

        spectral = cluster.SpectralClustering(n_clusters=K,
                                              eigen_solver='arpack',
                                              affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1

        return grp, L

    def best_map(self, L1, L2):

        # L1 should be the groundtruth labels and L2 should be the clustering labels we got

        Label1 = np.unique(L1)
        nClass1 = len(Label1)
        Label2 = np.unique(L2)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))

        for i in range(nClass1):
            ind_cla1 = L1 == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = L2 == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)

        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        newL2 = np.zeros(L2.shape)

        for i in range(nClass2):
            newL2[L2 == Label2[i]] = Label1[c[i]]

        return newL2

