from torch_cluster import knn_graph
from torch_geometric.nn.conv import XConv,PPFConv        # PPF Conv is roto-translational equivariant
import torch.nn as nn
import torch

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


from torch_cluster import knn_graph
from torch_geometric.nn.conv import XConv,PPFConv        # PPF Conv is roto-translational equivariant
import torch.nn as nn
import torch

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing



class PointAutoEncoderV2(nn.Module):
    def __init__(self, num_points, latent_dim, num_features = 0):
        """
            Architecture from https://arxiv.org/pdf/1707.02392.pdf  Learning Representations and Generative Models for 3D Point Clouds
            Adapted from https://github.com/TropComplique/point-cloud-autoencoder

            Arguments: 
                num_points: integer describing number of points in each padded point cloud
                latent_dim: integer describing dimensionality of latent vector
        """
        super(PointAutoEncoderV2, self).__init__()

        pointwise_layers = []
        num_units = [num_features + 3, 64, 128, 128, 256, latent_dim]

        for n, m in zip(num_units[:-1], num_units[1:]):
            pointwise_layers.extend([
                nn.Conv1d(n, m, kernel_size=1, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True)
            ])

        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.num_features= num_features
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim//2, num_points)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim//2, num_points)
        )
        
        self.apply(self.weights_init_uniform)


    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.normal_(mean=0.0, std=0.12)
            m.bias.data.fill_(0)
        if classname.find('Conv1d')!= -1:
            m.weight.data.normal_(mean=0.0, std=0.12)


    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """
        x = x.unsqueeze(0)
        x = torch.permute(x, (0, 2, 1))
        b, _, num_points = x.size()
        x = self.pointwise_layers(x)  # shape [b, k, num_points]
        encoding = self.pooling(x)  # shape [b, k, 1]

        reconstructed_feature1 = self.mlp1(encoding.T).T
        reconstructed_feature2 = self.mlp2(encoding.T).T

        print(reconstructed_feature1.shape, reconstructed_feature2.shape)

        reconstructed_points = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        reconstructed_points = reconstructed_points.view(b, num_points, 3)
        
        reconstructed_x = torch.cat((reconstructed_points,reconstructed_feature1,reconstructed_feature2 ), axis=1)

        return encoding, reconstructed_x




class PointAutoEncoder(nn.Module):
    def __init__(self, num_points, latent_dim, num_features = 0):
        """
            Architecture from https://arxiv.org/pdf/1707.02392.pdf  Learning Representations and Generative Models for 3D Point Clouds
            Adapted from https://github.com/TropComplique/point-cloud-autoencoder

            Arguments: 
                num_points: integer describing number of points in each padded point cloud
                latent_dim: integer describing dimensionality of latent vector
        """
        super(PointAutoEncoder, self).__init__()

        pointwise_layers = []
        num_units = [num_features + 3, 64, 128, 128, 256, latent_dim]

        for n, m in zip(num_units[:-1], num_units[1:]):
            pointwise_layers.extend([
                nn.Conv1d(n, m, kernel_size=1, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True)
            ])

        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.num_features= num_features
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )
        self.apply(self.weights_init_uniform)


    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.normal_(mean=0.0, std=0.12)
            m.bias.data.fill_(0)
        if classname.find('Conv1d')!= -1:
            m.weight.data.normal_(mean=0.0, std=0.12)


    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """
        x = x.unsqueeze(0)
        x = torch.permute(x, (0, 2, 1))
        b, _, num_points = x.size()
        x = self.pointwise_layers(x)  # shape [b, k, num_points]
        encoding = self.pooling(x)  # shape [b, k, 1]

        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, num_points, 3 + self.num_features)

        return encoding, restoration





class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)
        print(input.shape)
        mess = self.mlp(input) 
        print(mess)
        return mess


class PointNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        
    def forward(self, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        return h

class XConvEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, kernel_size=10):
        super().__init__()

        torch.manual_seed(12345)
        
        self.conv1 = XConv(num_features,hidden_dim, dim=num_features, kernel_size= kernel_size, hidden_channels= num_features)
        self.conv2 = XConv(hidden_dim,hidden_dim, dim=num_features, kernel_size= kernel_size, hidden_channels= num_features)
        # self.conv2 = XConv(hidden_dim,hidden_dim, dim=num_features, kernel_size= kernel_size, hidden_channels= num_features)
        # self.conv2 = XConv(hidden_dim,hidden_dim, dim=num_features, kernel_size= kernel_size, hidden_channels= num_features)
        # self.conv2 = XConv(hidden_dim,hidden_dim, dim=num_features, kernel_size= kernel_size, hidden_channels= num_features)

        
    def forward(self, pos, batch):
        x = self.conv1(x=pos, pos=pos, batch=batch)
        x = x.relu()
        x = self.conv2(x=x, pos=pos, batch=batch)
        return x


class XConvDecoder(torch.nn.Module):
    def __init__(self, input_dim, out_features):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = XConv(input_dim, input_dim, dim=input_dim, kernel_size= 5, hidden_channels= input_dim)
        self.conv2 = XConv(input_dim,out_features, dim=input_dim, kernel_size= 5, hidden_channels= out_features)
        
    def forward(self, pos, batch):
        x = self.conv1(x=pos, pos=pos, batch=batch)
        x = x.relu()
        x = self.conv2(x=x, pos=pos,  batch=batch)
        return x
    
    
class LinearDecoder(torch.nn.Module):
    def __init__(self, input_dim, out_features, hidden_dim=16):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, out_features)

        
    def forward(self, z):
        x_ = self.linear1(z).relu()
        x_ = self.linear2(x_).relu()
        x_ = self.linear3(x_).relu()
        x_ = self.linear4(x_)
        return x_
    
    


class XConvAutoEncoder(torch.nn.Module):
    def __init__(self, num_features, num_points, hidden_dim=128, kernel_size=10):
        super().__init__()

        torch.manual_seed(12345)
        # self.encoder = nn.Sequential(
        #     XConv(num_features, 64, dim=3, kernel_size=10, hidden_channels=3),
        #     nn.ReLU(inplace=True),
        #     XConv(64, 128, dim=3, kernel_size=10, hidden_channels=3),
        #     nn.ReLU(inplace=True),
        #     XConv(128, 128, dim=3, kernel_size=10, hidden_channels=3),
        #     nn.ReLU(inplace=True),
        #     XConv(128, 256, dim=3, kernel_size=10, hidden_channels=3),
        # )
        self.conv1 = XConv(num_features, 64, dim=3, kernel_size=10, hidden_channels=3)
        self.conv2 = XConv(64, 128, dim=3, kernel_size=10, hidden_channels=3)
        self.conv3 = XConv(128, 128, dim=3, kernel_size=10, hidden_channels=3)
        self.conv4 = XConv(128, 256, dim=3, kernel_size=10, hidden_channels=3)
        self.conv5 = XConv(256, hidden_dim, dim=3, kernel_size=10, hidden_channels=3)

        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, 256, kernel_size=1, bias=False),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            # nn.BatchNorm1d(256),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )
        
        
    def forward(self, pos, batch):
        num_points = pos.shape[0]
        x = self.conv1(x=pos, pos=pos, batch=batch).relu()
        x = self.conv2(x=x, pos=pos, batch=batch).relu()
        x = self.conv3(x=x, pos=pos, batch=batch).relu()
        x = self.conv4(x=x, pos=pos, batch=batch).relu()
        x = self.conv5(x=x, pos=pos, batch=batch)
        encoding = self.pooling(x.T)  # shape [b, k, 1]
        decoded = self.decoder(encoding)
        restoration = decoded.view(3, num_points)

        # x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        # restoration = x.view(b, 3, num_points)

        return restoration.T
    
    
class LinearPointAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
 
    def forward(self, pos, batch):
        z = self.encode(pos, batch)
        x_ = self.decode(z)
        return x_
    
    def encode(self, pos, batch):
        encoded = self.encoder(pos, batch)
        return encoded
    
    def decode(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
