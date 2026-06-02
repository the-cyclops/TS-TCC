import torch
import numpy as np
# Normalized Temperature-scaled Cross Entropy
# this loss is for Contextual Contrasting
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        # mask to remove auto-similarity (c_i with c_i) and positive similarity (c_i with c_i+)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        # matrix of dimension 2batch_size since for each sample in the batch, we have 2 augmented views
        # in the matrix, the first batch_size rows correspond to weak and the next batch_size rows correspond to strong augmentations
        # 1. Main diagonal (Self-similarity matrix)
        # [1 0 0 0] -> weak 1 vs weak 1
        # [0 1 0 0] -> weak 2 vs weak 2
        # [0 0 1 0] -> strong 1   vs strong 1
        # [0 0 0 1] -> strong 2   vs strong 2
        diag = np.eye(2 * self.batch_size)
        
        # 2. Lower diagonal (Positive pairs: strong row vs weak col)
        # [0 0 0 0]
        # [0 0 0 0]
        # [1 0 0 0] -> strong 1 vs weak 1
        # [0 1 0 0] -> strong 2 vs weak 2
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        
        # 3. Upper diagonal (Positive pairs: weak row vs strong col)
        # [0 0 1 0] -> weak 1 vs strong 1
        # [0 0 0 1] -> weak 2 vs strong 2
        # [0 0 0 0]
        # [0 0 0 0]
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        
        # 4. Combined invalid elements (diag + l1 + l2)
        # [1 0 1 0] -> 1s represent self or positive pairs
        # [0 1 0 1]
        # [1 0 1 0]
        # [0 1 0 1]
        mask = torch.from_numpy((diag + l1 + l2))
        
        # 5. Logical inversion (1 - mask) to get negatives, converted to Boolean
        # [F T F T] -> True (T) isolates negative samples only
        # [T F T F] -> False (F) excludes self and positive pairs
        # [F T F T]
        # [T F T F]
        mask = (1 - mask).type(torch.bool)
        
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # positives shape: (2*batch_size, 1), each row is similarity score of a positive pair 
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # negatives shape: (2*batch_size, 2*batch_size-2), each row is similarity score of a negative pair
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # logits shape: (2*batch_size, 2*batch_size-1), first column is positive pair similarity
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        # label is 0 for the first column (positive pair) 
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()

        # Contrastive learning as a multi-class classification problem.
        # Since 'positives' is placed in the 1st column (index 0) and 'negatives' in the rest,
        # 'labels=0' forces CrossEntropy to maximize the positive pair while minimizing all negatives.
        # CrossEntropy automatically computes the InfoNCE loss (Positive / (Positive + Sum of Negatives))
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
