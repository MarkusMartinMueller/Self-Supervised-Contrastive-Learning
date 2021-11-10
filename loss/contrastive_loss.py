import torch
import torch.nn as nn
import math



class NTxentLoss(nn.Module):
    """
       NTXent Loss (Normalized Temperature-scaled Cross-entropy Loss) .
    """
    def __init__(self,temperature,):
        super(NTxent,self).__init__()
        self.temperature = temperature

    def forward(self,modality_s1,modality_s2):

