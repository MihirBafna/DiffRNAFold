import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_cluster import knn_graph
from torchvision.ops import MLP
from pytorch3d.loss import chamfer_distance
from . import utils
from .EquivariantGNN import EGNN


class EquivariantGraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim):
        super(EquivariantGraphEncoder, self).__init__()
        self.egnn = EGNN(in_node_nf=in_dim-3, hidden_nf=32, out_node_nf=32, in_edge_nf=edge_dim, n_layers=3)


    def forward(self, h, x, edge_index, edge_attr):
        latent_features, latent_points = self.egnn(h, x, edge_index, edge_attr)
        return latent_features, latent_points


class EquivariantGraphDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim):
        super(EquivariantGraphDecoder, self).__init__()
        self.egnn = EGNN(in_node_nf=32, hidden_nf=32, out_node_nf=out_dim-3, in_edge_nf=edge_dim, n_layers=3)


    def forward(self, h, x, edge_index, edge_attr):
        recon_features, recon_points = self.egnn(h, x, edge_index, edge_attr)
        return recon_features, recon_points
    
    def recon_loss_equivariant(self, x, x_, h, h_):
        mse = torch.nn.MSELoss()
        return chamfer_distance(x_.unsqueeze(0), x.unsqueeze(0))[0], mse(h_, h), mse(x_, x)



class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, 2 * embed_dim)
        self.conv2 = GCNConv(2 * embed_dim, embed_dim)
        # self.LatentExtractor = nn.Sequential(
        #                             nn.Linear(embed_dim, embed_dim//2),
        #                             nn.ReLU(),
        #                             nn.Linear(embed_dim//2, embed_dim//4),
        #                             nn.ReLU(),
        #                             nn.Linear(embed_dim//4, embed_dim//8),
        #                             nn.ReLU(),
        #                             nn.Linear(embed_dim//8, 1),
        #                             )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return  x
    
    # def forward(self, x, edge_index, edge_attr):
    #     x = self.conv1(x, edge_index).relu()
    #     x = self.conv2(x, edge_index).relu()
    #     x = self.LatentExtractor(x)
    #     return  x

  
class GraphDecoder(torch.nn.Module):
    def __init__(self, embed_dim, coord_dim=3):
        super(GraphDecoder, self).__init__()
        self.ReconDecoder = InnerProductDecoder()
        self.CoordinateExtractor = MLP(embed_dim, [embed_dim * 2, coord_dim], bias=True, dropout=0.1, inplace=False)
        # self.XCoordinateExtractor = nn.Sequential(
        #                                 nn.Linear(1, embed_dim//8),
        #                                 nn.ReLU(),
        #                                 nn.Linear(embed_dim//8, embed_dim//8),
        #                                 nn.ReLU(),
        #                                 nn.Linear(embed_dim//8, 1),
        #                             )
        # self.YCoordinateExtractor = nn.Sequential(
        #                         nn.Linear(1, embed_dim//8),
        #                         nn.ReLU(),
        #                         nn.Linear(embed_dim//8, embed_dim//8),
        #                         nn.ReLU(),
        #                         nn.Linear(embed_dim//8, 1),
        #                     )
        # self.ZCoordinateExtractor = nn.Sequential(
        #                     nn.Linear(1, embed_dim//8),
        #                     nn.ReLU(),
        #                     nn.Linear(embed_dim//8, embed_dim//8),
        #                     nn.ReLU(),
        #                     nn.Linear(embed_dim//8, 1),
        #                 )


    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        return self.ReconDecoder(z, edge_index, sigmoid)

    # def extract_coordinates(self, latent):
    #     X = self.XCoordinateExtractor(latent)
    #     Y = self.XCoordinateExtractor(latent)
    #     Z = self.XCoordinateExtractor(latent)
    #     return torch.cat((X,Y,Z), axis=1)

    def extract_coordinates(self, z):
        return self.CoordinateExtractor(z)
    
    def recon_coord_loss(self, z, y):
        # not using innerproduct decoder anymore
        reconstructed_coords = self.extract_coordinates(z)
        rmse = utils.RMSELoss()
        mse = nn.MSELoss()
        return chamfer_distance(reconstructed_coords.unsqueeze(0), y.unsqueeze(0))[0], mse(reconstructed_coords, y)
    


      

class InnerProductDecoder(torch.nn.Module):
   r"""The inner product decoder from the `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>` paper


   .. math::\sigma(\mathbf{Z}\mathbf{Z}^{\top}) where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent space produced by the encoder."""
  
   def forward(self, z: Tensor, edge_index: Tensor,
                   sigmoid: bool = True) -> Tensor:
           r"""Decodes the latent variables :obj:`z` into edge probabilities for
           the given node-pairs :obj:`edge_index`.


           Args:
               z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
               sigmoid (bool, optional): If set to :obj:`False`, does not apply
                   the logistic sigmoid function to the output.
                   (default: :obj:`True`)
           """
           value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
           return torch.sigmoid(value) if sigmoid else value


   def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
           r"""Decodes the latent variables :obj:`z` into a probabilistic dense
           adjacency matrix.


           Args:
               z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
               sigmoid (bool, optional): If set to :obj:`False`, does not apply
                   the logistic sigmoid function to the output.
                   (default: :obj:`True`)
           """
           adj = torch.matmul(z, z.t())
           return torch.sigmoid(adj) if sigmoid else adj




