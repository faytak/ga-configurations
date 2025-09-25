import torch

import torch.nn.functional as F
from torch import nn

# from glgenn.algebra.cliffordalgebraex import CliffordAlgebraQT
# from glgenn.engineer.metrics.metrics import Loss, MetricCollection
# from glgenn.engineer.trainer.trainer import Trainer
# from glgenn.layers.qtgp import QTGeometricProduct
# from glgenn.layers.qtlinear import QTLinear
# from glgenn.layers.qtnorm import QTNormalization

from algebra.cliffordalgebra_subspaces import CliffordAlgebraSubspaces
from layers.ga_linear import GALinear
from layers.ga_geometricproduct import GAGeometricProduct
from layers.ga_normalization import GANormalization
from layers.ga_silu import GASiLU
from engineer.metrics.metrics import Loss, MetricCollection


# for LG
from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear
from models.modules.mvlayernorm import MVLayerNorm
from models.modules.mvsilu import MVSiLU

import torch
from torch import nn


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class EMLP_GA(nn.Module): #---------------------------------------------------------
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        subspace_type = None,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers


        layers = []

        if subspace_type == "LG":
            for i in range(n_layers - 1):
                layers.append(
                    nn.Sequential(
                        MVLinear(self.algebra, in_features, hidden_features),
                        MVSiLU(self.algebra, hidden_features),
                        SteerableGeometricProductLayer(
                            self.algebra,
                            hidden_features,
                            normalization_init=normalization_init,
                            include_first_order=False,
                        ),
                        MVLayerNorm(self.algebra, hidden_features),
                    )
                )
                in_features = hidden_features

            # Add final layer.
            layers.append(
                nn.Sequential(
                    MVLinear(self.algebra, in_features, out_features),
                    MVSiLU(self.algebra, out_features),
                    SteerableGeometricProductLayer(
                        self.algebra,
                        out_features,
                        normalization_init=normalization_init,
                        include_first_order=False,
                    ),
                    MVLayerNorm(self.algebra, out_features),
                )
            )

        else:
            if "-" in subspace_type:
                subspace_type_list = subspace_type.split("-")
                subspace_type_list = ["Triple" if i == "Tr" else i for i in subspace_type_list]

                    # Add geometric product layers.
                for i in range(n_layers - 1):
                    layers.append(
                        nn.Sequential(
                            GALinear(self.algebra, in_features, out_features, group_type=subspace_type_list[1]),  #---------------------------------------------------------
                            GASiLU(self.algebra, hidden_features, group_type=subspace_type_list[2]), #---------------------------------------------------------
                            GAGeometricProduct( #---------------------------------------------------------
                                self.algebra,
                                out_features,
                                normalization_init=normalization_init,
                                include_first_order=False,
                                group_type=subspace_type_list[3]
                            ),
                            GANormalization(self.algebra, hidden_features, group_type=subspace_type_list[4]), #---------------------------------------------------------
                        )
                    )
                    in_features = hidden_features

                # Add final layer.
                layers.append(
                    nn.Sequential(
                        GALinear(self.algebra, in_features, out_features, group_type=subspace_type_list[5]), #---------------------------------------------------------
                        GASiLU(self.algebra, out_features, group_type=subspace_type_list[6]), #---------------------------------------------------------
                        GAGeometricProduct( #---------------------------------------------------------
                            self.algebra,
                            out_features,
                            normalization_init=normalization_init,
                            include_first_order=False,
                            group_type=subspace_type_list[7]
                        ),
                        GANormalization(self.algebra, out_features, group_type=subspace_type_list[8]),
                    )
                )
            
            else:
                # Add geometric product layers.
                for i in range(n_layers - 1):
                    layers.append(
                        nn.Sequential(
                            GALinear(self.algebra, in_features, out_features, group_type=subspace_type),  #---------------------------------------------------------
                            GASiLU(self.algebra, hidden_features, group_type=subspace_type), #---------------------------------------------------------
                            GAGeometricProduct( #---------------------------------------------------------
                                self.algebra,
                                out_features,
                                normalization_init=normalization_init,
                                include_first_order=False,
                                group_type=subspace_type
                            ),
                            GANormalization(self.algebra, hidden_features, group_type=subspace_type), #---------------------------------------------------------
                        )
                    )
                    in_features = hidden_features

                # Add final layer.
                layers.append(
                    nn.Sequential(
                        GALinear(self.algebra, in_features, out_features, group_type=subspace_type), #---------------------------------------------------------
                        GASiLU(self.algebra, out_features, group_type=subspace_type), #---------------------------------------------------------
                        GAGeometricProduct( #---------------------------------------------------------
                            self.algebra,
                            out_features,
                            normalization_init=normalization_init,
                            include_first_order=False,
                            group_type=subspace_type
                        ),
                        GANormalization(self.algebra, out_features, group_type=subspace_type),
                    )
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EGCL_GA(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
        subspace_type = None,
    ):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = EMLP_GA( #---------------------------------------------------------
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            subspace_type=subspace_type,
            normalization_init=normalization_init,
        )

        self.node_model = EMLP_GA( #---------------------------------------------------------
            algebra,
            self.in_features + self.out_features,
            self.hidden_features,
            self.out_features,
            subspace_type=subspace_type,
            normalization_init=normalization_init,
        )

    def message(self, h_i, h_j, edge_attr=None):
        if edge_attr is None:  # Unused.
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)

        h_msg = self.edge_model(input)
        return h_msg

    def aggregate(self, h_msg, segment_ids, num_segments):
        h_agg = unsorted_segment_mean(h_msg, segment_ids, num_segments=num_segments)
        return h_agg

    def update(self, h_agg, h, node_attr=None):
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)

        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h

        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        # Message
        rows, cols = edge_index

        h_i, h_j = h[rows], h[cols]

        h_msg = self.message(h_i, h_j, edge_attr)

        # Aggregate
        agg_h = self.aggregate(h_msg.flatten(1), rows, num_segments=len(h)).view(
            len(h), *h_msg.shape[1:]
        )

        # Update
        h = self.update(agg_h, h, node_attr)

        return h


class NBodyGNN_GA(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=28,
        out_features=1,
        edge_features_in=1,
        n_layers=3,
        normalization_init=0,
        residual=True,
        subspace_type = None,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.n_layers = n_layers

        if subspace_type == "LG":
            self.algebra = CliffordAlgebra((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)) # n=6
            self.embedding = MVLinear(
                self.algebra, in_features, hidden_features, subspaces=False
            ) #---------------------------------------------------------

        else:
            self.algebra = CliffordAlgebraSubspaces((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)) # n=6  
            if "-" in subspace_type:
                subspace_type_list = subspace_type.split("-")
                subspace_type_list = ["Triple" if i == "Tr" else i for i in subspace_type_list]
                if subspace_type_list[0] == "LG":
                    self.embedding = MVLinear(self.algebra, in_features, hidden_features, subspaces=False) #---------------------------------------------------------
                else:
                    self.embedding = GALinear(self.algebra, in_features, hidden_features, group_type=subspace_type_list[0]) #---------------------------------------------------------
            else:
                self.embedding = GALinear(self.algebra, in_features, hidden_features, group_type=subspace_type) #---------------------------------------------------------

        layers = []

        for i in range(0, n_layers):
            layers.append(
                EGCL_GA( #---------------------------------------------------------
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                    subspace_type=subspace_type,
                )
            )
        if subspace_type == "LG":
            self.projection = nn.Sequential(MVLinear(self.algebra, hidden_features, out_features),) #---------------------------------------------------------
        else:
            if "-" in subspace_type:
                subspace_type_list = subspace_type.split("-")
                subspace_type_list = ["Triple" if i == "Tr" else i for i in subspace_type_list]
                if subspace_type_list[9] == "LG":
                    self.projection = nn.Sequential(MVLinear(self.algebra, hidden_features, out_features),) #---------------------------------------------------------
                else:
                    self.projection = nn.Sequential(GALinear(self.algebra, hidden_features, out_features, group_type=subspace_type_list[9]),) #---------------------------------------------------------
            else:
                self.projection = nn.Sequential(GALinear(self.algebra, hidden_features, out_features, group_type=subspace_type),) #---------------------------------------------------------

        self.layers = nn.Sequential(*layers)

        self.train_metrics = MetricCollection({"loss": Loss(),})
        self.test_metrics = MetricCollection({"loss": Loss(),})

        self.monitor_metric = "loss"
        self.monitor_mode = "min"

    def _forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)

        h = self.projection(h)
        return h

    def forward(self, batch, step):
        loc, vel, edge_attr, charges, loc_end, edges = batch

        batch_size, n_nodes, _ = loc.size()

        loc_mean = loc - loc.mean(dim=1, keepdim=True)

        loc_mean = loc_mean.float().view(-1, *loc_mean.shape[2:])
        loc = loc.float().view(-1, *loc.shape[2:])
        vel = vel.float().view(-1, *vel.shape[2:])
        edge_attr = edge_attr.float().view(-1, *edge_attr.shape[2:])
        charges = charges.float().view(-1, *charges.shape[2:])
        loc_end = loc_end.float().view(-1, *loc_end.shape[2:])

        # Add batch to graph.
        batch_index = torch.arange(batch_size, device=loc_mean.device)
        edges = edges + n_nodes * batch_index[:, None, None]
        edges = tuple(edges.transpose(0, 1).flatten(1))

        edge_attr = edge_attr
        edge_attr = self.algebra.embed(edge_attr[..., None], (0,))

        invariants = charges
        invariants = self.algebra.embed(invariants, (0,))

        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.algebra.embed(xv, (1, 2, 3, 4, 5, 6)) # change here

        input = torch.cat([invariants[:, None], covariants], dim=1)

        loc_pred = self._forward(input, edges, edge_attr)

        loc_pred = loc_pred[..., 0, 1:7] # change here
        loc_pred = loc + loc_pred
        loss = F.mse_loss(loc_pred, loc_end, reduction="none").mean(dim=1)

        return loss.mean(), {"loss": loss}