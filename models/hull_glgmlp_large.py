# This file contains GLGENN model for Convex Hull Experiment 
# for the case when the number K of points generating the hull is large (e.g. 256 or 512).
# In the case of smaller K, please use GLGENN from hull_glgmlp.py

import torch.nn.functional as F
from torch import nn

# from glgenn.algebra.cliffordalgebraex import CliffordAlgebraQT
# from glgenn.engineer.metrics.metrics import Loss, MetricCollection
# from glgenn.layers.qtgp import QTGeometricProduct
# from glgenn.layers.qtlinear import QTLinear


from algebra.cliffordalgebra_subspaces import CliffordAlgebraSubspaces
from layers.ga_linear import GALinear
from layers.ga_geometricproduct import GAGeometricProduct
# from layers.ga_normalization import GANormalization
from engineer.metrics.metrics import Loss, MetricCollection

class ConvexHullGLGMLP_large(nn.Module):
    def __init__(
        self,
        n,
        in_features=NUM_POINTS, # number of points for convex hull
        hidden_features=HIDDEN_FEATURES, # we take 128 for K=256 and K=512
        out_features=1,
        num_layers=4,
        subspace_type = "Q"
    ):
        super().__init__()

        self.n = n
        # self.algebra = CliffordAlgebraQT(
        #     (1.0,) * self.n
        # )

        self.algebra = CliffordAlgebraSubspaces(
            (1.0,) * self.n
        )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers

        self.net = nn.Sequential(
            GALinear(self.algebra, in_features, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GALinear(self.algebra, hidden_features, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
            GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

        self.train_metrics = MetricCollection({"loss": Loss()})
        self.test_metrics = MetricCollection({"loss": Loss()})

    def _forward(self, x):
        return self.net(x)

    def forward(self, batch, step):
        points, products = batch
        input = self.algebra.embed_grade(points, 1)

        y = self._forward(input)

        y = y.norm(dim=-1)
        y = self.mlp(y).squeeze(-1)

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(y, products, reduction="none")

        return loss.mean(), {
            "loss": loss,
        }