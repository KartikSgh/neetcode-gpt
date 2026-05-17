import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        data = raw_dataset.split()
        X, Y = [], []
        s = set()
        n = len(data)
        for i in range(batch_size):
            st = torch.randint(n-context_length, (1, 1))
            while (st in s):
                st = torch.randint(n-context_length, (1, 1))
            s.add(st)
            X.append(data[st:st+context_length])
            Y.append(data[st+1:st+1+context_length])
        return (X, Y)

