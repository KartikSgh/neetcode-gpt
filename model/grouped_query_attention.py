import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        q = self.q_proj(x) # (B, seq, num_heads*head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = torch.repeat_interleave(k, repeats=(self.num_heads // self.num_kv_heads), dim=1)
        v = torch.repeat_interleave(v, repeats=(self.num_heads // self.num_kv_heads), dim=1)

        scaled_dot_p = ((q @ k.transpose(-2, -1))/ (self.head_dim**0.5))
        mask = torch.tril(torch.ones(T, T))
        scaled_dot_p = scaled_dot_p.masked_fill(mask == 0, float('-inf'))

        attention_score = torch.softmax(scaled_dot_p, dim=-1)
        attention = attention_score @ v
        attention = attention.transpose(1, 2).contiguous().view(B, T, -1)
        final_attention = self.output_proj(attention)
        return torch.round(final_attention, decimals = 4)
