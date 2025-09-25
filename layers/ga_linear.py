import math
import torch
from torch import nn
from abc import ABC, abstractmethod

from .ga_config import get_linear_config


class BaseLinear(nn.Module, ABC):
    def __init__(self, algebra, in_features, out_features, bias=True):
        """
        Base linear layer class.

        Args:
            algebra: Clifford algebra object
            in_features: number of input features
            out_features: number of output features
        """
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, self.weight_init_dim()))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter("bias", None)
            self.b_dims = ()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    @abstractmethod
    def weight_init_dim(self) -> int:
        """Subclasses return size of last dim in self.weight."""
        pass

    @abstractmethod
    def process_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Subclasses implement how weight should be repeated."""
        pass

    def forward(self, input):
        weight = self.process_weight(self.weight)
        result = torch.einsum("bm...i, nmi->bn...i", input, weight)
        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result


# -------------------------------
# Specializations


class GALinear(BaseLinear):
    """
    Unified linear layer supporting P, Q, A, B, and Triple group types.
    
    This class replaces the individual P, Q, A, B, and Triple linear classes
    by using a group_type parameter to determine the appropriate configuration.
    """
    
    def __init__(self, algebra, in_features, out_features, group_type, bias=True):
        """
        Args:
            algebra: Clifford algebra object
            in_features: number of input features
            out_features: number of output features
            group_type: one of 'P', 'Q', 'A', 'B', 'Triple'
            bias: whether to include bias
        """
        self.group_type = group_type
        self.config = get_linear_config(group_type)
        super().__init__(algebra, in_features, out_features, bias)

        self.register_buffer(
            "permutation",
             torch.as_tensor(getattr(self.algebra, self.config['perm_attr']), dtype=torch.long),
        )
        self.register_buffer(
            "dims_sizes",
            torch.tensor([getattr(self.algebra, d) for d in self.config['dims_attr']],
            dtype=torch.long),
        )


    def weight_init_dim(self):
        """Return size of last dim in self.weight based on group type."""
        return self.config['weight_init_dim']

    def process_weight(self, weight):
        """Process weight according to group type."""
        # dim_sizes = torch.tensor([getattr(self.algebra, d) for d in self.config['dims_attr']], 
                                # device=self.weight.device)
        return weight.repeat_interleave(self.dims_sizes, dim=-1)[:, :, self.permutation]
    

# -------------------------------
# utils

def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]
