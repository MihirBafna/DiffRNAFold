from tqdm import tqdm
import wandb
import torch


def train_model(model, optimizer, train_loader, epochs, criterion, val_loader, intermediate_save_path=None):
    mse = torch.nn.MSELoss()
    val_loss = 0
    val_mseloss = 0
    val_rec_loss = 0
    val_kl_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        mse_loss = 0
        model.train()
        with tqdm(train_loader) as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                optimizer.zero_grad()  # Clear gradients.
                # reconstructed = model(data.pos, data.batch)# Forward pass.
                reconstructed, mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z = model(data.pos)
                
                # calculate num nodes that are unpadded
                data_detached = data.detach()
                num_atoms = data_detached.atom_number.max() - data_detached.atom_number.min() + 1
                # print(reconstructed[:, :num_atoms, :].shape, data.pos.unsqueeze(0)[:, :num_atoms, :].shape)
                
                reconstruction_loss = criterion(reconstructed[:, :num_atoms, :], data.pos.unsqueeze(0)[:, :num_atoms, :], bidirectional=True)
                KLD_element_x = torch.pow(mu_x, 2) + torch.pow(
			        sigma_x, 2) - torch.log(1e-8 + torch.pow(sigma_x, 2)) - 1.0
                KLD_element_y = torch.pow(mu_y, 2) + torch.pow(
                    sigma_y, 2) - torch.log(1e-8 + torch.pow(sigma_y, 2)) - 1.0
                KLD_element_z = torch.pow(mu_z, 2) + torch.pow(
                    sigma_z, 2) - torch.log(1e-8 + torch.pow(sigma_z, 2)) - 1.0
                loss_KL = 0.5 * (torch.sum(KLD_element_x) + torch.sum(KLD_element_y) + torch.sum(KLD_element_z))

                # mseloss = mse(reconstructed[:, :num_atoms, 4:],  data.pos.unsqueeze(0)[:, :num_atoms, 4:])
                # loss += mseloss
                loss = reconstruction_loss + loss_KL
                mseloss = mse(reconstructed.detach(), data.pos.unsqueeze(0).detach())
                loss.backward()  # Backward pass.
                optimizer.step()  # Update model parameters.
                tepoch.set_postfix(loss=loss.item(), mse = mseloss.item())
                epoch_loss += loss.item()
                epoch_rec_loss += reconstruction_loss.item()
                epoch_kl_loss += loss_KL.item()
                mse_loss += mseloss.item()

  
        if epoch % 5 == 0 and val_loader is not None:
            val_loss = 0
            val_rec_loss = 0
            val_kl_loss = 0
            val_mseloss = 0
            model.eval()
            for valdata in val_loader:
                reconstructed, mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z = model(valdata.pos)
                KLD_element_x = torch.pow(mu_x, 2) + torch.pow(
			        sigma_x, 2) - torch.log(1e-8 + torch.pow(sigma_x, 2)) - 1.0
                KLD_element_y = torch.pow(mu_y, 2) + torch.pow(
                    sigma_y, 2) - torch.log(1e-8 + torch.pow(sigma_y, 2)) - 1.0
                KLD_element_z = torch.pow(mu_z, 2) + torch.pow(
                    sigma_z, 2) - torch.log(1e-8 + torch.pow(sigma_z, 2)) - 1.0
                loss_KL = (2 * torch.sum(KLD_element_x) + torch.sum(KLD_element_y) + torch.sum(KLD_element_z)).item()
                reconstruction_loss = criterion(reconstructed.detach(), data.pos.unsqueeze(0).detach()).item()
                val_loss += loss_KL + reconstruction_loss
                val_rec_loss += reconstruction_loss
                val_kl_loss += loss_KL
                val_mseloss += mse(reconstructed.detach(), data.pos.unsqueeze(0).detach()).item()
        wandb.log({"epoch": epoch, "train_loss": epoch_loss/len(tepoch), "train_rec_loss": epoch_rec_loss/len(tepoch), "train_kl_loss": epoch_kl_loss/len(tepoch), "train_mseloss":mseloss/len(train_loader),
                "val_loss":val_loss/len(val_loader),  "val_rec_loss": val_rec_loss/len(val_loader), "val_kl_loss": val_kl_loss/len(val_loader), "val_mseloss":val_mseloss/len(val_loader)})
        
        if epoch == 100 and intermediate_save_path is not None:
            torch.save(model.state_dict(), intermediate_save_path[0])
        
        if epoch == 500 and intermediate_save_path is not None:
            torch.save(model.state_dict(), intermediate_save_path[1])
        
    return model