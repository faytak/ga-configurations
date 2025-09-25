import torch
from torch import nn

from .ga_config import get_normalization_config

EPS = 1e-6


class NormalizationBase(nn.Module):
    def __init__(self, algebra, features, norm_fn, dims_attr, perm_attr, init: float = 0):
        """
        Base normalization class.

        Args:
            algebra: Clifford algebra object
            features: number of features
            norm_fn: callable that takes input -> norms (e.g. algebra.norms_qt)
            dims_attr: list of attribute names for subspaces dims (e.g. ["dim_even", "dim_odd"])
            perm_attr: name of permutation attribute for the subspaces (e.g. "parity_weights_permutation")
            init: initialization constant for scaling parameter
        """
        super().__init__()
        self.algebra = algebra
        self.features = features
        self.norm_fn = norm_fn
        self.dims_attr = dims_attr
        self.perm_attr = perm_attr

        # learnable scaling per feature + per subspace
        self.a = nn.Parameter(torch.zeros(self.features, len(self.dims_attr)) + init)

        self.register_buffer(
            "permutation",
             torch.as_tensor(getattr(self.algebra, self.perm_attr), dtype=torch.long),
        )
        self.register_buffer(
            "dims_sizes",
            torch.tensor([getattr(self.algebra, d) for d in self.dims_attr],
            dtype=torch.long),
        )

    def forward(self, input):
        assert input.shape[1] == self.features

        # compute subspace norms
        norms = torch.cat(self.norm_fn(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1

        # repeat per basis element according to dim sizes
        # dim_sizes = torch.tensor([getattr(self.algebra, d) for d in self.dims_attr], device=norms.device)
        norms = norms.repeat_interleave(self.dims_sizes, dim=-1)

        # apply permutation
        # permutation = getattr(self.algebra, self.perm_attr)
        norms = norms[:, :, self.permutation]

        # normalize
        return input / (norms + EPS)


# -------------------------------
# Specializations

class GANormalization(NormalizationBase):
    """
    Unified normalization layer supporting P, Q, A, B, and Triple group types.
    
    This class replaces the individual P, Q, A, B, and Triple normalization classes
    by using a group_type parameter to determine the appropriate configuration.
    """
    
    def __init__(self, algebra, features, group_type, init: float = 0):
        """
        Args:
            algebra: Clifford algebra object
            features: number of features
            group_type: one of 'P', 'Q', 'A', 'B', 'Triple'
            init: initialization constant for scaling parameter
        """
        config = get_normalization_config(group_type)
        norm_fn = getattr(algebra, config['norm_fn_attr'])
        
        super().__init__(
            algebra=algebra,
            features=features,
            norm_fn=norm_fn,
            dims_attr=config['dims_attr'],
            perm_attr=config['perm_attr'],
            init=init,
        )
