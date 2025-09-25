import torch.nn.functional as F
from torch import nn

# from glgenn.algebra.cliffordalgebraex import CliffordAlgebraQT
# from glgenn.engineer.metrics.metrics import Loss, MetricCollection
# from glgenn.layers.qtgp import QTGeometricProduct
# from glgenn.layers.qtlinear import QTLinear

from algebra.cliffordalgebra_subspaces import CliffordAlgebraSubspaces
from layers.ga_linear import GALinear
from layers.ga_geometricproduct import GAGeometricProduct
from engineer.metrics.metrics import Loss, MetricCollection

class OnGLGMLP(nn.Module):
    def __init__(
        self,
        n,
        ymean, 
        ystd,
        subspace_type,
        output_qtgp=8,
        hidden_mlp_1=580,
        hidden_mlp_2=580,
        if_mlp=True
    ):
        super().__init__()
        self.n = n
        if subspace_type == "LG":
            from algebra.cliffordalgebra import CliffordAlgebra
            from models.modules.gp import SteerableGeometricProductLayer
            from models.modules.linear import MVLinear

            self.algebra = CliffordAlgebra(
                (1.0,) * self.n
            )
            self.gp = nn.Sequential(
                MVLinear(self.algebra, 2, output_qtgp, subspaces=False),
                SteerableGeometricProductLayer(self.algebra, output_qtgp, include_first_order=False),
                SteerableGeometricProductLayer(self.algebra, output_qtgp, include_first_order=False),
                SteerableGeometricProductLayer(self.algebra, output_qtgp, include_first_order=False),
                SteerableGeometricProductLayer(self.algebra, output_qtgp, include_first_order=False),
            )

        else:
            self.algebra = CliffordAlgebraSubspaces(
                (1.0,) * self.n
            )
            if "-" in subspace_type:
                subspace_type_list = subspace_type.split("-")
                subspace_type_list = ["Triple" if i == "Tr" else i for i in subspace_type_list]

                self.gp = nn.Sequential(
                    GALinear(self.algebra, 2, output_qtgp, group_type=subspace_type_list[0]),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type_list[0], include_first_order=False),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type_list[1], include_first_order=False),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type_list[2], include_first_order=False),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type_list[3], include_first_order=False),
                )
            else:
                self.gp = nn.Sequential(
                    GALinear(self.algebra, 2, output_qtgp, group_type=subspace_type),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type, include_first_order=False),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type, include_first_order=False),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type, include_first_order=False),
                    GAGeometricProduct(self.algebra, output_qtgp, group_type=subspace_type, include_first_order=False),
                )
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(output_qtgp, hidden_mlp_1),
        #     nn.ReLU(),
        #     nn.Linear(hidden_mlp_1, hidden_mlp_2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_mlp_2, 1),
        # )
        self.no_mlp = nn.Linear(output_qtgp, 1)
        self.if_mlp = if_mlp


        self.train_metrics = MetricCollection({"loss": Loss()})
        self.test_metrics = MetricCollection({"loss": Loss()})

        self.ymean = ymean
        self.ystd = ystd

        self.monitor_metric = "loss"
        self.monitor_mode = "min"

    # def forward(self, batch, step):
    #     points, products = batch

    #     points = points.view(len(points), 2, self.n)
    #     input = self.algebra.embed_grade(points, 1)

    #     if self.if_mlp:
    #         y = self.mlp(self.qtgp(input)[..., 0])
    #     else: 
    #         y = self.no_mlp(self.qtgp(input)[..., 0])
    
    #     normalized_y = y * self.ystd + self.ymean
    #     normalized_products = products * self.ystd + self.ymean

    #     assert y.shape == products.shape, breakpoint()
    #     loss = F.mse_loss(normalized_y, normalized_products.float(), reduction="none")
    #     return loss.mean(), {
    #         "loss": loss,
    #     }

    def forward(self, batch, step):
        n = self.algebra.n_subspaces - 1   # dimension d

        points, products = batch # points: [B, 2*n + 4]

        # split: r1, r2 (vectors), sc1, sc2 (scalars), ps1, ps2 (pseudoscalars)
        vectors = points[:, :2*n].view(len(points), 2, n)   # [B, 2, d]
        scalars = points[:, 2*n:2*n+2]                       # [B, 2]
        pseudoscalars = points[:, 2*n+2:]                    # [B, 2]

        # embed vectors as grade-1
        input_vectors = self.algebra.embed_grade(vectors, grade=1)  # [B, 2, ...]

        # pseudoscalars → grade 8
        input_pseudoscalars = self.algebra.embed_grade(pseudoscalars.unsqueeze(-1), grade=8)      # [B, 2, ...]

        # scalars → grade 0
        input_scalars = self.algebra.embed_grade(scalars.unsqueeze(-1), grade=0)  # [B, 2, ...]

        # combine everything into one multivector
        input = input_vectors + input_pseudoscalars + input_scalars

        y = self.no_mlp(self.gp(input)[..., 0])

        normalized_y = y * self.ystd + self.ymean
        normalized_products = products * self.ystd + self.ymean

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(normalized_y, normalized_products.float(), reduction="none")
        return loss.mean(), {
            "loss": loss,
        }


