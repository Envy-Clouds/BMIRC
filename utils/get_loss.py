import torch.nn.functional as F
import torch.nn as nn
import torch


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class Sim_Loss(nn.Module):

    def __init__(self, tokens_dim):
        super().__init__()
        self.fc_x = nn.Linear(tokens_dim, 128, device=torch.device('cuda'))
        self.fc_y = nn.Linear(tokens_dim, 128, device=torch.device('cuda'))
        self.sim = CMD()

    def forward(self, x, y, sim_type):
        x_feat = F.normalize(self.fc_x(x), dim=-1)
        y_feat = F.normalize(self.fc_y(y), dim=-1)
        if sim_type == 'cmd':
            sim_loss = self.sim(x_feat, y_feat, 5)
        elif sim_type == 'kl':
            q = nn.functional.softmax(x_feat, dim=-1)
            p = nn.functional.softmax(y_feat, dim=-1)
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            sim_loss = kl_loss(q.log(), p)
        elif sim_type == 'cos':
            sim_loss = (1 - F.cosine_similarity(x_feat, y_feat)).mean()
        else:
            sim_loss = torch.zeros(1, device=torch.device('cuda'))
        return sim_loss


def get_sim_loss(contrastive_tokens):
    x = contrastive_tokens['ecg']
    y = contrastive_tokens['pcg']
    loss_sim = Sim_Loss(x.shape[-1])
    loss = loss_sim(x, y, '')
    return loss
