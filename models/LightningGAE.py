'''
#-------------------------------------------------------------------------------------------------------------#
|  PyG implementation of GAE, but converted to be a pytorch-lightning module                                  |
|  https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#GAE |
#-------------------------------------------------------------------------------------------------------------#
'''


import lightning.pytorch as pl
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import wandb
torch.autograd.set_detect_anomaly(True)




EPS = 1e-15
MAX_LOGSTD = 10


class LightningGAE(pl.LightningModule):
  
   r"""The Graph Auto-Encoder model from the `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>` paper based on user-defined encoder and decoder models.


   Args:
       encoder (torch.nn.Module): The encoder module.
       decoder (torch.nn.Module, optional): The decoder module. If set to
           :obj:`None`, will default to the
           :class:`torch_geometric.nn.models.InnerProductDecoder`.
           (default: :obj:`None`)
   """
  
   def __init__(self, encoder: Module, decoder: Module):
       super().__init__()
       self.encoder = encoder
       # self.decoder = InnerProductDecoder() if decoder is None else decoder
       self.decoder = decoder
       LightningGAE.reset_parameters(self)


   def reset_parameters(self):
           r"""Resets all learnable parameters of the module."""
           reset(self.encoder)
           reset(self.decoder)


   def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
           r"""Alias for :meth:`encode`."""
           return self.encoder(*args, **kwargs)


   def encode(self, *args, **kwargs) -> Tensor:
           r"""Runs the encoder and computes node-wise latent variables."""
           return self.encoder(*args, **kwargs)


   def decode(self, *args, **kwargs) -> Tensor:
           r"""Runs the decoder and computes edge probabilities."""
           return self.decoder(*args, **kwargs)


   def recon_loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Optional[Tensor] = None) -> Tensor:
           r"""Given latent variables :obj:`z`, computes the binary cross
           entropy loss for positive edges :obj:`pos_edge_index` and negative
           sampled edges.


           Args:
               z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
               pos_edge_index (torch.Tensor): The positive edges to train against.
               neg_edge_index (torch.Tensor, optional): The negative edges to
                   train against. If not given, uses negative sampling to
                   calculate negative edges. (default: :obj:`None`)
           """
           pos_loss = -torch.log(
               self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()


           if neg_edge_index is None:
               neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
           neg_loss = -torch.log(1 -
                               self.decoder(z, neg_edge_index, sigmoid=True) +
                               EPS).mean()


           return pos_loss + neg_loss




   def test(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
       r"""Given latent variables :obj:`z`, positive edges
       :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
       computes area under the ROC curve (AUC) and average precision (AP)
       scores.


       Args:
           z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
           pos_edge_index (torch.Tensor): The positive edges to evaluate
               against.
           neg_edge_index (torch.Tensor): The negative edges to evaluate
               against.
       """
       from sklearn.metrics import average_precision_score, roc_auc_score


       pos_y = z.new_ones(pos_edge_index.size(1))
       neg_y = z.new_zeros(neg_edge_index.size(1))
       y = torch.cat([pos_y, neg_y], dim=0)


       pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
       neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
       pred = torch.cat([pos_pred, neg_pred], dim=0)


       y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
      


       return roc_auc_score(y, pred), average_precision_score(y, pred)
      
      
   def decode_and_evaluate(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index:  Optional[Tensor] = None):
      
       from sklearn.metrics import average_precision_score, roc_auc_score


       pos_y = z.new_ones(pos_edge_index.size(1))
       if neg_edge_index is None:
           neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
       neg_y = z.new_zeros(neg_edge_index.size(1))
       y = torch.cat([pos_y, neg_y], dim=0)


       pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)


       neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
       pred = torch.cat([pos_pred, neg_pred], dim=0)
      
       pos_loss = -torch.log(pos_pred + EPS).mean()
       neg_loss = -torch.log(1 - neg_pred + EPS).mean()
       recon_loss = pos_loss + neg_loss
              


       y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
       return recon_loss, roc_auc_score(y, pred), average_precision_score(y, pred)
      
       # return recon_loss
      


   def configure_optimizers(self):
       # optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
       optimizer = torch.optim.Adam(self.parameters(), lr=0.000005)
       return optimizer




   # EGNN encoder/decoder
  
   def training_step(self, train_batch, batch_idx):
       node_features, edge_index, edge_features, node_positions = train_batch.x[:,:-3], train_batch.edge_index, train_batch.edge_attr, train_batch.x[:,-3:]
       latent_feats, latent_points = self.encoder(node_features,node_positions, edge_index, edge_features)
       recon_feats, recon_points = self.decoder(node_features,node_positions, edge_index, edge_features)
       self.log_dict({"train_total_loss":train_total_loss, "train_rmsd":rmsd})
       wandb.log({"epoch":self.trainer.current_epoch,"train_total_loss":train_total_loss, "train_rmsd":rmsd})
       return train_total_loss
  


   def validation_step(self, val_batch, batch_idx):
       node_features, edge_index, edge_features, node_positions = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.x[:,-3:]
       z = self.encoder(node_features, edge_index, edge_features)       
       val_recon_coord_loss, rmsd = self.decoder.recon_coord_loss(z, node_positions)
       val_total_loss = val_recon_coord_loss
       self.log_dict({"val_total_loss":val_total_loss, "val_rmsd":rmsd})
       wandb.log({"epoch":self.trainer.current_epoch,"val_total_loss":val_total_loss, "val_rmsd":rmsd})




   # only chamfer loss
  
   # def training_step(self, train_batch, batch_idx):
   #     node_features, edge_index, edge_features, node_positions = train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.x[:,-3:]
   #     z = self.encoder(node_features, edge_index, edge_features)
   #     train_recon_coord_loss, rmsd = self.decoder.recon_coord_loss(z, node_positions)
   #     train_total_loss =  train_recon_coord_loss + rmsd
   #     # train_total_loss =  train_recon_coord_loss
   #     self.log_dict({"train_total_loss":train_total_loss, "train_rmsd":rmsd})
   #     wandb.log({"epoch":self.trainer.current_epoch,"train_total_loss":train_total_loss, "train_rmsd":rmsd})
   #     return train_total_loss
  


   # def validation_step(self, val_batch, batch_idx):
   #     node_features, edge_index, edge_features, node_positions = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.x[:,-3:]
   #     z = self.encoder(node_features, edge_index, edge_features)       
   #     val_recon_coord_loss, rmsd = self.decoder.recon_coord_loss(z, node_positions)
   #     val_total_loss = val_recon_coord_loss
   #     self.log_dict({"val_total_loss":val_total_loss, "val_rmsd":rmsd})
   #     wandb.log({"epoch":self.trainer.current_epoch,"val_total_loss":val_total_loss, "val_rmsd":rmsd})
      


   # def training_step(self, train_batch, batch_idx):
   #     node_features, edge_index, edge_features, node_positions = train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.x[:,-3:]
   #     z = self.encoder(node_features, edge_index, edge_features)
   #     train_recon_edge_loss, train_auroc, train_ap = self.decode_and_evaluate(z, edge_index[:, (edge_index < node_features.shape[0]).any(axis=0)])
   #     train_recon_coord_loss, rmsd = self.decoder.recon_coord_loss(z, node_positions)
   #     train_total_loss = train_recon_edge_loss + train_recon_coord_loss
   #     self.log_dict({"train_total_loss":train_total_loss, "train_auroc":train_auroc, "train_ap":train_ap, "train_recon_edge_loss":train_recon_edge_loss})
   #     wandb.log({"epoch":self.trainer.current_epoch,"train_total_loss":train_total_loss,"train_recon_edge_loss":train_recon_edge_loss,"train_auroc":train_auroc, "train_ap":train_ap, "train_rmsd":rmsd})
   #     return train_total_loss
  


   # def validation_step(self, val_batch, batch_idx):
   #     node_features, edge_index, edge_features, node_positions = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.x[:,-3:]
   #     z = self.encoder(node_features, edge_index, edge_features)       
   #     val_recon_edge_loss, val_auroc, val_ap = self.decode_and_evaluate(z, edge_index[:, (edge_index < node_features.shape[0]).any(axis=0)])
   #     val_recon_coord_loss, rmsd = self.decoder.recon_coord_loss(z, node_positions)
   #     val_total_loss = val_recon_edge_loss + val_recon_coord_loss
   #     self.log_dict({"val_total_loss":val_total_loss, "val_auroc":val_auroc, "val_ap":val_ap, "val_recon_edge_loss":val_recon_edge_loss})
   #     wandb.log({"epoch":self.trainer.current_epoch,"val_total_loss":val_total_loss,"val_recon_edge_loss":val_recon_edge_loss, "val_auroc":val_auroc, "val_ap":val_ap, "val_rmsd":rmsd})
      
      
      
