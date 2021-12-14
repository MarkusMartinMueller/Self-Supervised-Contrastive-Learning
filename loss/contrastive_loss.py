import torch
import torch.nn as nn
import math
from torch.nn import functional as F



class NTxentLoss(nn.Module):
    """
       NTXent Loss (Normalized Temperature-scaled Cross-entropy Loss) .
    """

    def __init__(self, temperature, device, eps=1e-6):
        super(NTxentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.device = device

    def forward(self,modality_s1,modality_s2):
        """



        modality_1: [batch_size, dim]
        modality_2: [batch_size, dim]

        # gather representations in case of distributed training
        # modality_1_dist: [batch_size * world_size, dim]
        # modality_2_dist: [batch_size * world_size, dim]
               Implementation taken from:
               https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py

               p - positive pair
               n - negative pair
               sim - cosine similarity
               e - Euler's number

               ix - value x of input feature vector i
               tx - value x of input feature vector t

                               Similarities matrix: exp(sim(i, y))
                                    +--+--+--+--+--+--+--+
                                    |  |i1|i2|i3|t1|t2|t3|
                Modality            +--+--+--+--+--+--+--+
                Features            |i1|e |n |n |p |n |n |
               +--+  +--+           +--+--+--+--+--+--+--+
               |i1|  |t1|           |i2|n |e |n |n |p |n |
               +--+  +--+           +--+--+--+--+--+--+--+
               |i2|  |t2|  ------>  |i3|n |n |e |n |n |p |
               +--+  +--+           +--+--+--+--+--+--+--+
               |i3|  |t3|           |t1|p |n |n |e |n |n |
               +--+  +--+           +--+--+--+--+--+--+--+
                                    |t2|n |p |n |n |e |n |
                                    +--+--+--+--+--+--+--+
                                    |t3|n |n |p |n |n |e |
                                    +--+--+--+--+--+--+--+

               :param out_1: input feature vector i
               :param out_2: input feature vector t
               :return: NTXent loss
               """


        modality_s1 = F.normalize(modality_s1)
        modality_s2 = F.normalize(modality_s2)

        out = torch.cat([modality_s1, modality_s2], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]


        cov = torch.mm(out, out.t().contiguous())

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(self.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(modality_s1 * modality_s2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss


if __name__ == "__main__":

    nxt = NTxentLoss(1)
    inputs_s1 = torch.randn((1, 128))
    inputs_s2 = torch.randn((1, 128))

    loss = nxt(inputs_s1, inputs_s2)

    print(loss)