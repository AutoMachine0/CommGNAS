import torch
import warnings
warnings.filterwarnings("ignore")
from commgnas.model.stack_gcn_encoder.gcn_encoder import GcnEncoder
from commgnas.model.logger import gnn_architecture_performance_save,\
                                  test_performance_save
from commgnas.dynamic_configuration import optimizer_getter,  \
                                           loss_getter, \
                                           evaluator_getter, \
                                           downstream_task_model_getter
from torch_geometric.utils import dropout_adj
from sklearn.preprocessing import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StackGcn(object):

    def __init__(self,
                 graph_data,
                 downstream_task_type="node_classification",
                 downstream_task_parameter={},
                 supervised_learning=True,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1,
                 gnn_architecture=['gcn', 'sum',  1, 128, 'relu', 'gcn', 'sum', 1, 64, 'linear'],
                 gnn_drop_out=0.6,
                 train_epoch=100,
                 train_epoch_test=100,
                 bias=True,
                 early_stop=False,
                 early_stop_patience=10,
                 opt_type="adam",
                 opt_parameter_dict={"learning_rate": 0.005, "l2_regularization_strength": 0.0005},
                 loss_type="nll_loss",
                 val_evaluator_type="accuracy",
                 test_evaluator_type=["accuracy", "precision", "recall", "f1_value"]):

        self.graph_data = graph_data
        self.downstream_task_type = downstream_task_type
        self.downstream_task_parameter = downstream_task_parameter
        self.supervised_learning = supervised_learning
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.gnn_architecture = gnn_architecture
        self.gnn_drop_out = gnn_drop_out
        self.train_epoch = train_epoch
        self.train_epoch_test = train_epoch_test
        self.bias = bias
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.opt_type = opt_type
        self.opt_parameter_dict = opt_parameter_dict
        self.loss_type = loss_type
        self.val_evaluator_type = val_evaluator_type
        self.test_evaluator_type = test_evaluator_type
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gnn_model = GcnEncoder(self.gnn_architecture,
                                    self.graph_data.num_features,
                                    dropout=self.gnn_drop_out,
                                    bias=self.bias).to(self.device)

        self.optimizer = optimizer_getter(self.opt_type,
                                          self.gnn_model,
                                          self.opt_parameter_dict)

        self.loss = loss_getter(self.loss_type)

        self.val_evaluator = evaluator_getter(self.val_evaluator_type)

        self.downstream_task_model = downstream_task_model_getter(self.downstream_task_type,
                                                                  self.downstream_task_parameter,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)

    def fit(self, mode='search'):

        x = self.graph_data.test_x
        edge_index = self.graph_data.test_edge_index

        print("Staring self supervised training")
        # construct random x1, x2, Edge1, Edge2

        drop_edge_rate1 = 0.2
        drop_edge_rate2 = 0.4

        x1 = x
        x2 = x

        edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate1)[0]
        edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate2)[0]

        # combine x1, x2 to a whole graph dataset
        x1_node_nums = x1.shape[0]

        # increase the second graph edge index and prepare
        # for combination of edge_index_1 and edge_index_2
        edge_index_2[0] = edge_index_2[0] + x1_node_nums
        edge_index_2[1] = edge_index_2[1] + x1_node_nums

        # combine x1 and x2
        train_x = torch.cat([x1, x2], dim=0)

        # combine edge_index_1 and edge_index_2:
        train_edge_index = torch.cat([edge_index_1, edge_index_2], dim=-1)

        train_loss_value = 0.0
        first_self_supervised_loss = 0.0

        for epoch in range(1, self.train_epoch + 1):

            node_embedding_matrix = self.gnn_model(train_x, train_edge_index)

            # split node_embedding_matrix to z1, z2
            node_embedding_matrix_ = node_embedding_matrix.view(2, self.graph_data.node_nums, -1)

            z1 = node_embedding_matrix_[0]
            z2 = node_embedding_matrix_[1]

            train_loss = self.loss.function(z1, z2)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # obtain the train loss value for recording
            # the first epoch self-supervised loss value
            # and the last epoch self-supervised loss value
            train_loss_value = train_loss.item()

            if epoch == 1:
                first_self_supervised_loss = train_loss_value

            if epoch % 10 == 0:
                print("epoch: ", epoch, " train loss value: ", train_loss_value)

        if mode == "test":

            pass

        else:
            # the last epoch self-supervised loss value
            final_self_supervised_loss = train_loss_value

            # self-supervised loss change
            self_supervised_loss_change = abs(first_self_supervised_loss - final_self_supervised_loss)

            Z = self.gnn_model(self.graph_data.test_x, self.graph_data.test_edge_index)

            # self.downstream_task_model represent the node self-representation learning.
            Z = torch.tensor(normalize(Z.cpu().detach().numpy())).to(device)
            self_expressive_loss_change = self.downstream_task_model(Z,
                                                                     None,
                                                                     mode="test")

            print("self_supervised_loss_change:", self_supervised_loss_change)
            print("self_expressive_loss_change:", self_expressive_loss_change)

            feedback = self_supervised_loss_change * self_expressive_loss_change
            print("feedback:", feedback)

            gnn_architecture_performance_save(self.gnn_architecture, feedback, self.graph_data.data_name)

            return feedback

    def evaluate(self):

        self.train_epoch = self.train_epoch_test

        self.fit(mode="test")

        self.downstream_task_model = downstream_task_model_getter("node_cluster",
                                                                  self.downstream_task_parameter,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)

        self.gnn_model.eval()

        Z = self.gnn_model(self.graph_data.test_x, self.graph_data.test_edge_index)

        acc, nmi, f1, y_pred_label, S = self.downstream_task_model(Z,
                                                                   None,
                                                                   mode="test")
        test_performance_dict = {"accuracy": acc,
                                 "NMI": nmi,
                                 "F1": f1}

        hyperparameter_dict = self.downstream_task_parameter

        test_performance_save(self.gnn_architecture,
                              test_performance_dict,
                              hyperparameter_dict,
                              self.graph_data.data_name)


if __name__=="__main__":
   pass