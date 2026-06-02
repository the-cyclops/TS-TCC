import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer


# Temporal contrasting module
# Calculate TC loss and return the projected features for contrastive loss calculation
class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        # timestep is K in the paper, which is the number of future steps we want to predict and contrast with. We set K=3 in our experiments.
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        # Non-linear projection head to map context c_t into a lower-dimensional space for Contextual Contrasting downstream
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features (weak) are (batch_size, #channels, seq_len)
        # seq_len is T  in the paper, and both are > K (timestep) 
        seq_len = z_aug1.shape[2]
        # Transpose to (batch_size, seq_len, #channels) for Transformer input compatibility
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]

        # randomly pick time stamps t to consider as the final time step for transformer 
        # from 0 to t will be the history, from t+1 to t+K will be the future
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  #t_samples + timestep < seq_len

        # Noise contrastive estimation (InfoNCE), is the TC loss
        nce = 0 

        # encode_samples shape: (timestep, batch, #channels) -> stores future true samples from strong view
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)

        # forward_seq shape: (batch_size, t_samples + 1, #channels) -> history context from Weak view
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        # pred shape: (timestep, batch, #channels) -> predicted future steps based on history context c_t
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        # Compute cross-view InfoNCE loss for Temporal Contrasting
        for i in np.arange(0, self.timestep):
            # total shape: (batch, batch) -> similarity matrix between Strong future actuals and Weak predicted futures
            # this is similarity matrix since the similarity metric is the dot product 
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))

            # 1. lsoftmax computes row-wise (in the batch) log-probabilities for EVERY cell in the matrix.
            #    For cell (i, j), the numerator is exp(similarity) and denominator is the sum of exp across row i.
            # 2. torch.diag isolates the correct positive pairs where row index equals column index (match bewteen predicted future and real future).
            # 3. torch.sum aggregates these values across the batch to maximize the correct matches.
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        # NCE loss is negative log likelihood, so we take negative and average over the batch and time steps
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)