import torch
import warnings
warnings.filterwarnings('always')
from sklearn.metrics import accuracy_score

class Evaluator:

    def function(self, y_predict, y_ture):
        _, y_predict = torch.max(y_predict, dim=1)
        y_predict = y_predict.to("cpu").detach().numpy()
        y_ture = y_ture.to("cpu").detach().numpy()
        accuracy = accuracy_score(y_ture, y_predict)
        return accuracy