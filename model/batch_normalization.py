import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            y = (x - mean)/np.sqrt(var+eps)
            y = y*gamma+beta
            running_mean = (1-momentum)*running_mean+momentum*mean
            running_var = (1-momentum)*running_var+momentum*var
            return np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4)
        else:
            y = (x-running_mean)/np.sqrt(running_var+eps)
            y = y*gamma+beta
            return np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4)
