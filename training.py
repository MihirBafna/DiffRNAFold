from tqdm import tqdm
import wandb



def train_model(model, optimizer, train_loader, epochs, criterion, val_loader):
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        with tqdm(train_loader) as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                optimizer.zero_grad()  # Clear gradients.
                # reconstructed = model(data.pos, data.batch)# Forward pass.
                _,reconstructed = model(data.pos)
                loss = criterion(reconstructed, data.pos.unsqueeze(0))
                loss.backward()  # Backward pass.
                optimizer.step()  # Update model parameters.
                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

        val_loss = 0
        if epoch % 5 == 0:
            model.eval()
            for valdata in val_loader:
                _,reconstructed = model(valdata.pos)
                val_loss += criterion(reconstructed, data.pos.unsqueeze(0)).item()
            
        wandb.log({"epoch": epoch, "train_loss": epoch_loss/len(tepoch), "val_loss":val_loss/len(val_loader)})
        
    return model