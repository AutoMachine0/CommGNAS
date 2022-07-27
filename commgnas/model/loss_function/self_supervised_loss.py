import torch
import torch.nn.functional as F

class Loss:

    def __init__(self, tau=0.4):
        self.tau = tau

    def function(self, z1, z2):
        loss = self.self_supervised_loss_(z1, z2)
        return loss

    def self_supervised_loss_(self,
                              z1,
                              z2,
                              mean=True):

        l1 = self.loss_calculate(z1, z2)
        l2 = self.loss_calculate(z2, z1)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()

        return loss

    def loss_calculate(self, z1, z2):

        device = z1.device

        num_nodes = z1.size(0)

        indices = torch.arange(0, num_nodes).to(device)
        rand_indices = torch.randperm(num_nodes).to(device)

        ordered_mask = indices[0:num_nodes]
        random_mask = rand_indices[0:num_nodes]

        self_sim = torch.exp(self.cos_sim(z1[ordered_mask], z1[random_mask])/self.tau)
        between_sim = torch.exp(self.cos_sim(z1[ordered_mask], z2[random_mask])/self.tau)

        loss = torch.log(self_sim.sum(1) + between_sim.sum(1)) - \
               (F.normalize(z1[ordered_mask]) * F.normalize(z2[ordered_mask])).sum(1) / self.tau

        return loss

    def cos_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())