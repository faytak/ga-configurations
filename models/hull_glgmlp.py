# This file contains GLGENN model for Convex Hull Experiment 
# for the case when the number K of points generating the hull is small (e.g. 16).
# In the case of large K, please use GLGENN from hull_glgmlp_large.py

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

from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear

# from algebra.cliffordalgebraex import CliffordAlgebraQT
# from layers.qtgp import QTGeometricProduct
# from layers.qtlinear import QTLinear

# from layers.ga_linear_F import QLinear
# from layers.ga_geometricproduct_F import QGeometricProduct
# from layers.ga_normalization_F import QNormalization

class ConvexHullGLGMLP(nn.Module):
    def __init__(
        self,
        n,
        in_features=16,
        hidden_features=32,
        out_features=1,
        num_layers=4,
        subspace_type = "Q"
    ):
        super().__init__()

        self.n = n

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        # self.algebra = CliffordAlgebraQT(
        #     (1.0,) * self.n
        # )

        if subspace_type == "LG":
            from algebra.cliffordalgebra import CliffordAlgebra
            from models.modules.gp import SteerableGeometricProductLayer
            from models.modules.linear import MVLinear

            self.algebra = CliffordAlgebra(
                (1.0,) * self.n
            )
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.out_features = out_features
            self.num_layers = num_layers

            self.net = nn.Sequential(
                MVLinear(self.algebra, in_features, hidden_features, subspaces=False),
                SteerableGeometricProductLayer(
                    self.algebra, hidden_features, include_first_order=True
                ),
                SteerableGeometricProductLayer(
                    self.algebra, hidden_features, include_first_order=True
                ),
                SteerableGeometricProductLayer(
                    self.algebra, hidden_features, include_first_order=True
                ),
                SteerableGeometricProductLayer(
                    self.algebra, hidden_features, include_first_order=True
                ),
            )

        else:
            self.algebra = CliffordAlgebraSubspaces(
                (1.0,) * self.n
            )

            if "-" in subspace_type:
                from models.modules.gp import SteerableGeometricProductLayer

                subspace_type_list = subspace_type.split("-")
                subspace_type_list = ["Triple" if i == "Tr" else i for i in subspace_type_list]

                layers = []
                if subspace_type_list[0] == "LG":
                    layers.append(MVLinear(self.algebra, in_features, hidden_features, subspaces=False))
                else:
                    layers.append(GALinear(self.algebra, in_features, hidden_features, group_type=subspace_type_list[0]))

                for i in range(1, len(subspace_type_list)):
                    include_first_order = True if i == 1 else False
                    if subspace_type_list[i] == "LG":
                        layers.append(SteerableGeometricProductLayer(self.algebra, hidden_features, include_first_order=include_first_order))
                    else:
                        layers.append(GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type_list[i], include_first_order=include_first_order))
                    
                self.net = nn.Sequential(*layers)
                # self.net = nn.Sequential(
                #     GALinear(self.algebra, in_features, hidden_features, group_type=subspace_type_list[0]),
                #     GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type_list[1], include_first_order=True),
                #     GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type_list[2], include_first_order=False),
                #     GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type_list[3], include_first_order=False),
                #     GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type_list[4], include_first_order=False),
                # )

            else:
                self.net = nn.Sequential(
                    GALinear(self.algebra, in_features, hidden_features, group_type=subspace_type),
                    GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type, include_first_order=True),
                    GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type, include_first_order=False),
                    GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type, include_first_order=False),
                    GAGeometricProduct(self.algebra, hidden_features, group_type=subspace_type, include_first_order=False),
                )

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

        self.train_metrics = MetricCollection({"loss": Loss()})
        self.test_metrics = MetricCollection({"loss": Loss()})

        self.monitor_metric = "loss"
        self.monitor_mode = "min"

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

        return loss.mean(), {"loss": loss,}
