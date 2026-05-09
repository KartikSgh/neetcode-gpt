import torch
import torch.nn as nn
from typing import List
import numpy as np


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        ans = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    y = x.numpy()
                    ans.append(round(np.sum(np.all(y==0, axis=0))/len(y[0]), 4))
        return ans


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        t = [False for i in range(5)]
        prev = -1
        temp = True
        for i, f in enumerate(dead_fractions):
            if f>0.5:
                t[0] = True
                break
            if i==0 and f>0.3:
                t[1] = True
            if f<=prev:
                temp = False
            prev = f
            if i==len(dead_fractions)-1 and f<=0.1:
                temp = False
        if t[0]:
            return 'use_leaky_relu'
        if t[1]:
            return 'reinitialize'
        if temp:
            return 'reduce_learning_rate'
        return 'healthy'
            
