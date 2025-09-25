import math
import torch
from torch import nn

from .ga_linear import GALinear
from .ga_normalization import GANormalization
from .ga_config import get_geometric_product_config


class GeometricProductBase(nn.Module):
    def __init__(self,
                 algebra,
                 features,
                 normalization_cls,
                 linear_cls,
                 product_paths_attr,
                 dims_attr,
                 perm_attr,
                 normalization_init=0,
                 include_first_order=True):
        """
        Base geometric product class.

        Args:
            algebra: Clifford algebra object
            features: number of features (here in_features==out_features; for different number of input and output features, see ga_fcgeometricproduct.py)
            normalization_cls: normalization approach class (e.g. PNormalization)
            linear_cls: linear combination parameterization approach class (e.g. PLinear)
            product_paths_attr: attribute name for mask for geometric product paths that depend on a group (e.g. parity_geometric_product_paths)
            dims_attr: list of attribute names for subspaces dims (e.g. ["dim_even", "dim_odd"])
            perm_attr: name of permutation attribute for the subspaces (e.g. "parity_weights_permutation")
            normalization_init: initialization value for normalization
            include_first_order: whether to add residual-like connections
        """

        super().__init__()
        self.algebra = algebra
        self.features = features
        self.product_paths = getattr(algebra, product_paths_attr)  # e.g. algebra.parity_geometric_product_paths
        self.dims_attr = dims_attr  # e.g. ["dim_even", "dim_odd"] or ["dim_0", "dim_1", "dim_2", "dim_3"]
        self.perm_attr = perm_attr # e.g. parity_weights_permutation
        self.include_first_order = include_first_order

        # normalization
        if normalization_init is not None:
            self.normalization = normalization_cls(algebra, features, normalization_init)
        else:
            self.normalization = nn.Identity()

        # linear transform of input
        self.linear_right = linear_cls(algebra, features, features, bias=False)

        # one more linear transform of input (in case we want a kind of residual connection after nonlinearity)
        if include_first_order:
            self.linear_left = linear_cls(algebra, features, features, bias=True)

        # parameters
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))
        
        self.register_buffer(
            "permutation",
            torch.as_tensor(getattr(self.algebra, self.perm_attr), dtype=torch.long),
        )
        self.register_buffer(
            "dims_sizes",
            torch.tensor([getattr(self.algebra, d) for d in self.dims_attr],
            dtype=torch.long),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.algebra.dim + 1))

    def _get_weight(self):
        # [features, num_subspaces, num_subspaces, num_subspaces]
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )

        mask = self.product_paths
        if isinstance(mask, torch.Tensor) and mask.device != self.weight.device:
            mask = mask.to(self.weight.device)
        weight[:, mask] = self.weight   # [features, 4, 4, 4]

        # get dim sizes for repeat_interleave
        # dim_sizes = torch.tensor([getattr(self.algebra, d) for d in self.dims_attr], device=self.weight.device)

        # expand to full [features, 2**n, 2**n, 2**n]
        # permutation = getattr(self.algebra, self.perm_attr)
        weight_repeated = (
            weight
            .repeat_interleave(self.dims_sizes, dim=-1)[:, :, :, self.permutation]
            .repeat_interleave(self.dims_sizes, dim=-2)[:, :, self.permutation, :]
            .repeat_interleave(self.dims_sizes, dim=-3)[:, self.permutation, :, :]
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
        weight = self._get_weight()

        # parameterize geometric product of input and normalization(linear1(input)), and then add linear2(input):
        if self.include_first_order:
            return (self.linear_left(input) + torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)) / math.sqrt(2)

        # parameterize geometric product of input and normalization(linear(input)):
        else:
          return torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)


# -------------------------------
# Specializations


class GAGeometricProduct(GeometricProductBase):
    """
    Unified geometric product layer supporting P, Q, A, B, and Triple group types.
    
    This class replaces the individual P, Q, A, B, and Triple geometric product classes
    by using a group_type parameter to determine the appropriate configuration.
    """
    
    def __init__(
        self,
        algebra,
        features,
        group_type,
        normalization_init=0,
        include_first_order=True,
    ):
        """
        Args:
            algebra: Clifford algebra object
            features: number of features (here in_features==out_features; for different number of input and output features, see ga_fcgeometricproduct.py)
            group_type: one of 'P', 'Q', 'A', 'B', 'Triple'
            normalization_init: initialization value for normalization
            include_first_order: whether to add residual-like connections
        """
        config = get_geometric_product_config(group_type)
        
        # Create lambda functions for the classes to avoid circular imports
        def create_normalization_cls(alg, feat, init):
            return GANormalization(alg, feat, group_type, init)
        
        def create_linear_cls(alg, in_feat, out_feat, bias=True):
            return GALinear(alg, in_feat, out_feat, group_type, bias)
        
        super().__init__(
            algebra=algebra,
            features=features,
            normalization_cls=create_normalization_cls,
            linear_cls=create_linear_cls,
            product_paths_attr=config['product_paths_attr'],
            dims_attr=config['dims_attr'],
            perm_attr=config['perm_attr'],
            normalization_init=normalization_init,
            include_first_order=include_first_order,
        )