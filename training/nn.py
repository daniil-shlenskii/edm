import abc
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, Optional

class LearnableTimesteps(nn.Module, abc.ABC):
    # returns: [0, ..., 1]
    def __init__(self, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
    
    @abc.abstractmethod
    def forward(self, image: Optional[Tensor]=None, label: Optional[Tensor]=None):
        raise NotImplementedError

class LearableTimestepsDefault(LearnableTimesteps):
    def __init__(self, n_steps: int):
        super().__init__(n_steps)
        self.v = nn.Parameter(torch.ones(n_steps - 1))
        self.register_buffer("the_one", torch.ones(1))

    def forward(self, image: Tensor=None, label: Tensor=None) -> Tensor:
        timesteps = torch.cumsum(torch.softmax(torch.cat([self.v, self.the_one]), dim=0), dim=0)
        # print(f"{timesteps = }")
        return timesteps