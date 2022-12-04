from tqdm import tqdm
import wandb
import torch


def train_model(model, optimizer, train_loader, epochs, criterion, val_loader, intermediate_save_path=None):
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    val_loss = 0
    val_mseloss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        mse_loss = 0
        model.train()
        with tqdm(train_loader) as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                optimizer.zero_grad()  # Clear gradients.
                # reconstructed = model(data.pos, data.batch)# Forward pass.
                _,reconstructed = model(data.pos)
                
                # calculate num nodes that are unpadded
                data_detached = data.detach()
                num_atoms = data_detached.atom_number.max() - data_detached.atom_number.min() + 1
                # print(reconstructed[:, :num_atoms, :].shape, data.pos.unsqueeze(0)[:, :num_atoms, :].shape)
                
                chamferloss = criterion(reconstructed[:, :num_atoms, :], data.pos.unsqueeze(0)[:, :num_atoms, :])
                celoss = ce(reconstructed[:, :num_atoms, 4:],  data.pos.unsqueeze(0)[:, :num_atoms, 4:])
                
                loss = chamferloss + celoss
                mseloss = mse(reconstructed.detach(), data.pos.unsqueeze(0).detach())
                loss.backward()  # Backward pass.
                optimizer.step()  # Update model parameters.
                tepoch.set_postfix(loss=loss.item(), chamfer=chamferloss.item(), ce=celoss.item(), mse = mseloss.item())
                
                epoch_loss += loss.item()
                mse_loss += mseloss.item()

  
        if epoch % 5 == 0 and val_loader is not None:
            val_loss = 0
            val_mseloss = 0
            model.eval()
            for valdata in val_loader:
                _,reconstructed = model(valdata.pos)
                val_loss += criterion(reconstructed, data.pos.unsqueeze(0)).item()
                val_mseloss += mse(reconstructed.detach(), data.pos.unsqueeze(0).detach()).item()
        wandb.log({"epoch": epoch, "train_loss": epoch_loss/len(tepoch), "val_loss":val_loss/len(val_loader),  "train_mseloss":mseloss/len(train_loader), "val_mseloss":val_mseloss/len(val_loader)})
        
        if epoch == 100 and intermediate_save_path is not None:
            torch.save(model.state_dict(), intermediate_save_path[0])
        
        if epoch == 500 and intermediate_save_path is not None:
            torch.save(model.state_dict(), intermediate_save_path[1])
        
    return model