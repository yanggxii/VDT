from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F

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
        return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

        
    

class DiffLoss(nn.Module):
    def __init__(self, batch_size):
        super(DiffLoss, self).__init__()
        self.batch_size = batch_size
    
    def forward(self, input1, input2, input3):
        margin = 1.0
        
        sd_si_dist = F.pairwise_distance(input1, input3)
        td_ti_dist = F.pairwise_distance(input2, input3)
        
        diff_loss = F.relu(margin - sd_si_dist).mean() + F.relu(margin - td_ti_dist).mean()

        return diff_loss