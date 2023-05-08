import numpy as np
import torch.nn as nn
import torch

def rmsd(pred, truth):
    return np.linalg.norm(pred - truth) / np.sqrt(len(truth))

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss