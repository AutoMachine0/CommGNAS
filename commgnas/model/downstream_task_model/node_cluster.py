import torch
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn import cluster
from scipy.sparse.linalg import svds
from torch.nn.parameter import Parameter
from sklearn.preprocessing import normalize

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
                node_embedding_matrix,
                batch_x_index,
                mode="train"):

        Z = torch.tensor(normalize(node_embedding_matrix.cpu().detach().numpy())).to(self.device)

        print("\n Starting self representation training !")
        S = self.self_representation(Z, self.data.num_labels)

        print("\n Performance estimation of spectral clustering on the similarity matrix !")
        scores = self.spectral_cluster(S, self.data.test_y, self.data.num_labels)
        print(" Ac:", scores[0], "NMI:", scores[1], "F1:", scores[2])

        accuracy = scores[0]
        NMI = scores[1]
        F1 = scores[2]
        y_pred_label = scores[3]

        return accuracy, NMI, F1, y_pred_label, S

    def self_representation(self, Z, n_class):

        max_epoch = self.downstream_task_parameter['se_epochs']
        alpha = self.downstream_task_parameter['se_loss_reg']
        patience = self.downstream_task_parameter['patience']
        best_loss = 1e9
        bad_count = 0
        best_C = 0.0

        for epoch in range(max_epoch):

            self.semodel.train()
            self.seoptimizer.zero_grad()

            C, CZ = self.semodel(Z)

            se_loss = torch.norm(Z - CZ)
            reg_loss = torch.norm(C)
            loss = se_loss + alpha * reg_loss
            loss.backward()
            train_loss_value = loss.item()

            if epoch % 10 == 0:
                print("self representation learning train epoch: ", epoch, " train loss value: ", train_loss_value)

            self.seoptimizer.step()

            if loss.item() < best_loss:
                best_C = C
                bad_count = 0
                best_loss = loss.item()

            else:
                bad_count += 1
                if bad_count == patience:
                    break

        C = best_C.cpu().detach().numpy()
        S = self.similarity_matrix_computation(C, n_class, 4)

        return S

    def similarity_matrix_computation(self, C, K, d):

        # C: coefficient matrix,
        # K: number of clusters,
        # d: dimension of each subspace

        C_star = 0.5 * (C + C.T)
        r = min(d * K + 1, C_star.shape[0] - 1)
        U, Sig, _ = svds(C_star, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        Sig_sqrt = np.sqrt(Sig[::-1])
        Sig_sqrt = np.diag(Sig_sqrt)
        R_star = U.dot(Sig_sqrt)
        R_star = normalize(R_star, norm='l2', axis=1)
        R_star = R_star.dot(R_star.T)
        R_star = R_star * (R_star > 0)
        S = np.abs(R_star)
        S = 0.5 * (S + S.T)
        S = S / S.max()

        return S

    def spectral_cluster(self, S, test_true_y, n_class):

        y_prediction = self.cluster(S, n_class)
        print("Spectral clustering done.. finding fest match based on Kuhn-Munkres")
        scores = self.err_rate(test_true_y.detach().cpu().numpy(), y_prediction)
        return scores

    def err_rate(self, test_true_y, y_prediction):

        y_pred = self.best_match(test_true_y, y_prediction)
        acc = metrics.accuracy_score(test_true_y, y_pred)
        nmi = metrics.normalized_mutual_info_score(test_true_y, y_pred)
        f1_macro = metrics.f1_score(test_true_y, y_pred, average='macro')

        return [acc, nmi, f1_macro, y_pred]

    def cluster(self, S, K):

        # S: similarity matrix,
        # K: number of clusters,
        # d: dimension of each subspace

        spectral = cluster.SpectralClustering(n_clusters=K,
                                              eigen_solver='arpack',
                                              affinity='precomputed',
                                              assign_labels='discretize')

        output = spectral.fit_predict(S) + 1
        return output

    def best_match(self, L1, L2):

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


class SelfExpr(torch.nn.Module):

    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        self.C_ = Parameter(torch.FloatTensor(n, n).uniform_(0, 0.01))

    def forward(self, Z):

        C = self.C_ - torch.diag(torch.diagonal(self.C_))
        output = torch.mm(C, Z)

        return C, output