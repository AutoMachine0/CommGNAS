from munkres import Munkres
import numpy as np
from sklearn import metrics

class Evaluator:

    def function(self, y_predict, y_ture):

        y_true = y_ture.cpu().numpy()
        y_pred = self.best_map(y_true, y_predict)
        acc = metrics.accuracy_score(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        return acc, nmi, f1_macro

    def best_map(self, L1, L2):

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