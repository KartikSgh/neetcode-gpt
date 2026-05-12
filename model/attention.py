import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.W_k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.W_k(embedded)
        Q = self.W_q(embedded)
        V = self.W_v(embedded)
        attention_score = (Q@torch.transpose(K, 1, 2))/torch.sqrt(torch.tensor(self.attention_dim))

        mask = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1]))
        casual_attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
        score = nn.functional.softmax(casual_attention_score, dim=2)
        return torch.round(score @ V, decimals=4)


