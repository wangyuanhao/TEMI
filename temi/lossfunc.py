import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def REC_loss(pred_y, true_y):
    res = (pred_y - true_y) ** 2
    dim = true_y.shape[1]
    mask = torch.sum(torch.abs(true_y), dim=1) > 0
    loss = 1.0 / (torch.sum(mask) * dim) * torch.sum(torch.sum(res, dim=1) * mask)
    return loss

class PolarRegularization(nn.Module):
    def __init__(self, reduction, temperature=1):
        super(PolarRegularization, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, linear_prob, y):
        prob = torch.softmax(linear_prob, dim=1)
        prob_class0 = prob[:, 0]
        prob_class1 = prob[:, 1]
        diff = (2 * y - 1) * (prob_class1 - prob_class0) #- 0.1 * y
        loss_ = torch.exp((diff - 1) ** 2 / self.temperature)
        # loss_ = (diff - 1) ** 2
        if self.reduction == "mean":
            loss = loss_.sum() / len(y)
        else:
            loss = loss_.sum()

        return loss

class MarginDiff(nn.Module):
    def __init__(self, reduction, tau=1):
        super(MarginDiff, self).__init__()
        self.reduction = reduction
        self.tau = tau

    def forward(self, projection, y):
        if y.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        sim = torch.mm(projection, projection.T)
        same_mask = torch.eq(y.view(-1, 1), y.view(-1, 1).T).float() - torch.eye(len(y)).to(device)
        diff_mask = 1 - torch.eq(y.view(-1, 1), y.view(-1, 1).T).float()


        n = sim.shape[0]
        loss_ = []
        for i in range(n):
            if torch.sum(same_mask[i, :]) != 0:
                sim_same_class_ = sim[i, same_mask[i, :] == 1].min()
            else:
                sim_same_class_ = torch.tensor([0]).to(device)
            if torch.sum(diff_mask[i, :]) != 0:
                sim_diff_class_ = sim[i, diff_mask[i, :] == 1].max()
            else:
                sim_diff_class_ = torch.tensor([0]).to(device)

            loss_i_ = torch.maximum(torch.tensor([0]).to(device), sim_same_class_ +sim_diff_class_ + self.tau)
            loss_.append(loss_i_)
        loss__ = torch.stack(loss_)

        if self.reduction == "mean":
            loss = loss__.mean()
        else:
            loss = loss__.sum()

        return loss


class AdaptiveResidual(nn.Module):
    def __init__(self, beta, C):
        super(AdaptiveResidual, self).__init__()
        self.beta = beta
        self.C = C


    def forward(self, x, y):

        residual = x - y

        orth_loss = torch.norm(torch.mm(x, residual.T))
        mse_loss = torch.sum((torch.norm(residual, dim=1) - self.C)**2)
        n = x.shape[0]

        loss = 1 / n * (self.beta / 2 * mse_loss + orth_loss)

        return loss




