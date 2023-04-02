import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv,GATConv
from torch_cluster import knn_graph
from torchvision.ops import MLP
from pytorch3d.loss import chamfer_distance


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GATConv(in_dim, 2 * embed_dim, heads = 4, edge_dim=edge_dim) 
        self.conv2 = GATConv(2 * embed_dim, embed_dim, heads = 4, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index,edge_attr).relu()
        return self.conv2(x, edge_index,edge_attr)
    
    
class GraphDecoder(torch.nn.Module):
    def __init__(self, embed_dim, coord_dim=3):
        super(GraphDecoder, self).__init__()
        self.ReconDecoder = InnerProductDecoder()
        self.CoordinateExtractor = MLP(embed_dim, [embed_dim * 2], coord_dim, bias=True, dropout=0.1)

    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        return self.ReconDecoder(z, edge_index, sigmoid)

    def extract_coordinates(self, z):
        return self.CoordinateExtractor(z)
    
    def recon_coord_loss(self, z, y):
        return chamfer_distance(self.extract_coordinates(z), y)
    
    
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

