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

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable



class PointVAE(nn.Module):
	"""
	Adapted from: https://github.com/YiruS/PTC_VAE/tree/33cbebbd3f3567ef201338b8cb8cb8bf5568b2ad
	"""
	def __init__(self, num_points, latent_dim, num_features = 0, epsilon=1e-6):
		super(PointVAE, self).__init__()
		self.input_dim = num_features + 3
		self.zdim = latent_dim
		self.npts = num_points
		self.epsilon = epsilon

		## encoder ##
		self.E_conv1 = torch.nn.Conv1d(self.input_dim, 128, 1)  # 64
		self.E_conv2 = torch.nn.Conv1d(128, 128, 1)  # 64
		self.E_conv3 = torch.nn.Conv1d(128, 256, 1)  # 128
		self.E_conv4 = torch.nn.Conv1d(256, 512, 1)  # 512


		self.E_fcm = torch.nn.Linear(512, self.zdim)
		self.E_fcv = torch.nn.Linear(512, self.zdim)

		self.E_fcm_x = torch.nn.Linear(512, self.zdim)
		self.E_fcv_x = torch.nn.Linear(512, self.zdim)

		self.E_fcm_y = torch.nn.Linear(512, self.zdim)
		self.E_fcv_y = torch.nn.Linear(512, self.zdim)

		self.E_fcm_z = torch.nn.Linear(512, self.zdim)
		self.E_fcv_z = torch.nn.Linear(512, self.zdim)

		## deconder ##
		self.D_fc1 = torch.nn.Linear(self.zdim, 256)
		self.D_fc2 = torch.nn.Linear(256, 1024)
		self.D_fc3 = torch.nn.Linear(1024, num_points * (3 + num_features))

		## deconder along each axis (x,y,z) ##
		self.D_fc1_x = torch.nn.Linear(self.zdim, 256)
		self.D_fc1_y = torch.nn.Linear(self.zdim, 256)
		self.D_fc1_z = torch.nn.Linear(self.zdim, 256)

		self.D_fc2_x = torch.nn.Linear(256, num_points)
		self.D_fc2_y = torch.nn.Linear(256, num_points)
		self.D_fc2_z = torch.nn.Linear(256,num_points)

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.leakyrelu = nn.LeakyReLU(0.2)

		self.apply(self.weights_init_uniform)

	def encoder(self, x):
		x = x.transpose(1, 2)  # BxCxN
		x = self.relu(self.E_conv1(x))
		x = self.relu(self.E_conv2(x))
		x = self.relu(self.E_conv3(x))
		x = self.E_conv4(x)

		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 512)

		mu_x = self.E_fcm_x(x)
		sigma_x = self.E_fcv_x(x)
		stddev_x = self.epsilon + F.softplus(sigma_x)

		mu_y = self.E_fcm_y(x)
		sigma_y = self.E_fcv_y(x)
		stddev_y = self.epsilon + F.softplus(sigma_y)

		mu_z = self.E_fcm_z(x)
		sigma_z = self.E_fcv_z(x)
		stddev_z = self.epsilon + F.softplus(sigma_z)

		return mu_x, stddev_x, mu_y, stddev_y, mu_z, stddev_z

	def decoder(self, z):
		x = self.relu(self.D_fc1(z))
		x = self.relu(self.D_fc2(x))
		x = self.D_fc3(x)
		x = self.tanh(x)
		return x

	def decoder_each_axis(self, z_x, z_y, z_z):
		ptx = self.relu(self.D_fc1_x(z_x))
		ptx = self.D_fc2_x(ptx)

		pty = self.relu(self.D_fc1_y(z_y))
		pty = self.D_fc2_y(pty)

		ptz = self.relu(self.D_fc1_z(z_z))
		ptz = self.D_fc2_z(ptz)
		x = torch.cat((ptx.unsqueeze(2),pty.unsqueeze(2),ptz.unsqueeze(2)), dim=2)
		return x


	def reparameterize_gaussian(self, mu, sigma):
		# std = torch.exp(0.5 * logvar)
		# eps = torch.cuda.FloatTensor(std.size()).normal_()
		# eps = Variable(eps)
		# # eps = torch.rand_like(std, device=logvar.device)
		# # eps = Variable(torch.randn(std.size(), dtype=torch.float, device=std.device))
		# return eps.mul(std).add_(mu)  # mu + std * eps
		z = mu + sigma * torch.randn_like(mu, device=sigma.device)
		z = Variable(z)
		return z


	def forward(self, x):
		z_mu_x, z_sigma_x, z_mu_y, z_sigma_y, z_mu_z, z_sigma_z = self.encoder(x)
		z_x = self.reparameterize_gaussian(z_mu_x, z_sigma_x)
		z_y = self.reparameterize_gaussian(z_mu_y, z_sigma_y)
		z_z = self.reparameterize_gaussian(z_mu_z, z_sigma_z)

		ptc = self.decoder_each_axis(z_x, z_y, z_z)
		return ptc, z_mu_x, z_sigma_x, z_mu_y, z_sigma_y, z_mu_z, z_sigma_z

	def weights_init_uniform(self, m):
		classname = m.__class__.__name__
		# for every Linear layer in a model..
		if classname.find('Linear') != -1:
			# apply a uniform distribution to the weights and a bias=0
			m.weight.data.normal_(mean=0.0, std=0.12)
			m.bias.data.fill_(0)
		if classname.find('Conv1d')!= -1:
			m.weight.data.normal_(mean=0.0, std=0.12)



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
        x = torch.permute(x, (0, 2, 1))
        b, _, num_points = x.size()
        x = self.pointwise_layers(x)  # shape [b, k, num_points]
        encoding = self.pooling(x)  # shape [b, k, 1]

        reconstructed_feature1 = self.mlp1(encoding.T).T
        reconstructed_feature2 = self.mlp2(encoding.T).T

        print(reconstructed_feature1.shape, reconstructed_feature2.shape)

        reconstructed_points = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        reconstructed_points = reconstructed_points.view(b, num_points, 3)
        
        reconstructed_x = torch.cat((reconstructed_points,reconstructed_feature1,reconstructed_feature2 ), axis=2)

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
			nn.Conv1d(256, num_points * (3 + num_features), kernel_size=1)
		)
		
		self.apply(self.weights_init_uniform)
		# self.weights_init_uniform(self.pointwise_layers)
		# self.weights_init_uniform(self.decoder)


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
