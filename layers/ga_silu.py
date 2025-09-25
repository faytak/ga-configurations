import torch
from torch import nn

from .ga_config import get_silu_config, get_silu_dims_tensor


class BaseSiLU(nn.Module):
    """
    Base class for GA SiLU activation functions.
    """

    def __init__(self, algebra, channels, num_subspaces, dims_tensor, perm_attr, norm_method):
        """
        Args:
            algebra: Clifford algebra object
            channels: number of input features/channels
            num_subspaces: number of parameter subspaces (including scalars)
            dims_tensor: tensor of dimensions of each subspace
            perm_attr: attribute name for the permutation mapping
            norm_method: method name for computing squared norms
        """
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.num_subspaces = num_subspaces
        self.dims_tensor = dims_tensor
        self.perm_attr = perm_attr
        self.norm_method = norm_method

        # Learnable parameters for scaling and bias per subspace
        self.a = nn.Parameter(torch.ones(1, channels, num_subspaces))
        self.b = nn.Parameter(torch.zeros(1, channels, num_subspaces))

    def _get_norms(self, input):
        """Get squared norms using the specified norm method."""
        norm_func = getattr(self.algebra, self.norm_method)
        return norm_func(input)

    def forward(self, input):
        # Compute norms for each subspace
        norms = self._get_norms(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)

        # Apply learned scaling and bias
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b

        # Expand to full algebra using the appropriate permutation
        norms = norms.repeat_interleave(self.dims_tensor.to(self.a.device), dim=-1)
        permutation = getattr(self.algebra, self.perm_attr)
        norms = norms[..., permutation]

        # Apply SiLU activation
        return torch.sigmoid(norms) * input


# -------------------------------
# Specializations


class GASiLU(BaseSiLU):
    """
    Unified SiLU activation layer supporting P, Q, A, B, and Triple group types.
    
    This class replaces the individual P, Q, A, B, and Triple SiLU classes
    by using a group_type parameter to determine the appropriate configuration.
    """
    
    def __init__(self, algebra, channels, group_type):
        """
        Args:
            algebra: Clifford algebra object
            channels: number of input features/channels
            group_type: one of 'P', 'Q', 'A', 'B', 'Triple'
        """
        config = get_silu_config(group_type)
        dims_tensor = get_silu_dims_tensor(algebra, group_type)
        
        super().__init__(
            algebra=algebra,
            channels=channels,
            num_subspaces=config['num_subspaces'],
            dims_tensor=dims_tensor,
            perm_attr=config['perm_attr'],
            norm_method=config['norm_method']
        )


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Function code from:
    https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/modules/utils.py

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