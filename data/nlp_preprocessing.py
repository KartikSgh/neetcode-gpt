import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, p: List[str], n: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        l = []
        for i in p:
            l.extend(i.split())
        for i in n:
            l.extend(i.split())
        l = list(set(l))
        l.sort()
        print(l)
        _map = {l[i]:i+1 for i in range(len(l))}
        ans = []
        for i in p:
            t = [_map[j] for j in i.split()]
            ans.append(torch.tensor(t, dtype=torch.float32))
        for i in n:
            t = [_map[j] for j in i.split()]
            ans.append(torch.tensor(t, dtype=torch.float32))
        ans = torch.nn.utils.rnn.pad_sequence(ans, padding_value=0, batch_first=True)
        return ans

        
