from tqdm import tqdm
import wandb
import torch
import random
import plotly.express as px
import plotly.graph_objects as go

def train_model(model, optimizer, train_loader, epochs, criterion, val_loader, kl_coeff=1, intermediate_save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    val_loss = 0
    val_mseloss = 0
    val_rec_loss = 0
    val_kl_loss = 0
    model = model.to(device)
    rand_val_idx = random.randint(0, len(val_loader))
    rand_val_sub_idx = 0
    rand_idx = random.randint(0, len(train_loader))
    rand_sub_idx = 0
    plotly_steps = []
    val_plotly_steps = []
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        mse_loss = 0
        model.train()
        with tqdm(train_loader) as tepoch:
            # for data in tepoch:
            for idx, (data, _) in enumerate(tepoch):
                x = data.float().to(device)
                tepoch.set_description(f"Epoch: {epoch}")
                optimizer.zero_grad()  # Clear gradients.
                # reconstructed = model(data.pos, data.batch)# Forward pass.
                # data.pos = data.pos.view(batch_size, -1, model.input_dim)

                if type(model).__name__ == "PointVAE":
                    reconstructed, mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z = model(x)           
                    KLD_element_x = torch.pow(mu_x, 2) + torch.pow(
                        sigma_x, 2) - torch.log(1e-8 + torch.pow(sigma_x, 2)) - 1.0
                    KLD_element_y = torch.pow(mu_y, 2) + torch.pow(
                        sigma_y, 2) - torch.log(1e-8 + torch.pow(sigma_y, 2)) - 1.0
                    KLD_element_z = torch.pow(mu_z, 2) + torch.pow(
                        sigma_z, 2) - torch.log(1e-8 + torch.pow(sigma_z, 2)) - 1.0
                    loss_KL = kl_coeff * (torch.sum(KLD_element_x) + torch.sum(KLD_element_y) + torch.sum(KLD_element_z))
                    epoch_kl_loss += loss_KL.item()
                elif type(model).__name__ == "PointAutoEncoder" or type(model).__name__ == "PointAutoEncoderV2":
                    _, reconstructed = model(x)

                if idx == rand_idx:
                    plotly_steps.append(go.Frame(data=[go.Scatter3d(x=x[rand_sub_idx, :, 0].detach().cpu().numpy(), y=x[rand_sub_idx, :, 1].detach().cpu().numpy(), z=x[rand_sub_idx, :, 2].detach().cpu().numpy(), name="Ground Truth", mode='markers'),
                                    go.Scatter3d(x=reconstructed[rand_sub_idx, :, 0].detach().cpu().numpy(), y=reconstructed[rand_sub_idx, :, 1].detach().cpu().numpy(), z=reconstructed[rand_sub_idx, :, 2].detach().cpu().numpy(), name="Reconstructed", mode='markers')
                                   ]))
                    # plotly_steps.append({"epoch":epoch, "x": x[rand_sub_idx, :, 0].detach().cpu().numpy(), "y": x[rand_sub_idx, :, 1].detach().cpu().numpy(), "z": x[rand_sub_idx, :, 2].detach().cpu().numpy(), "type":  "Ground Truth"})
                    # plotly_steps.append({"epoch":epoch, "x": reconstructed[rand_sub_idx, :, 0].detach().cpu().numpy(), "y": reconstructed[rand_sub_idx, :, 1].detach().cpu().numpy(), "z": reconstructed[rand_val_sub_idx, :, 2].detach().cpu().numpy(), "type":  "Ground Truth"})
                    wandb.log({"epoch":epoch, "Ground Truth": wandb.Object3D({"type": "lidar/beta","points":x[rand_sub_idx, :, :]}), "Reconstructed": wandb.Object3D({"type": "lidar/beta","points":reconstructed[rand_sub_idx, :, :]})},commit=False)
    
                if type(model).__name__ == "PointVAE":
                    reconstruction_loss, _ = criterion(reconstructed[:, :, :], x[:, :, :])
                    loss = reconstruction_loss + loss_KL    
                elif type(model).__name__ == "PointAutoEncoder":
                    reconstruction_loss, _ = criterion(reconstructed[:, :, :], x[:, :, :])
                    loss = reconstruction_loss
                elif type(model).__name__ == "PointAutoEncoderV2":
                    reconstruction_loss = criterion(reconstructed[:, :, :3], data[:, :, :3])
                    celoss = ce(reconstructed[:, :, 3:],  data[:, :, 3:])
                    loss = reconstruction_loss + celoss

                
                mseloss = mse(reconstructed.detach(), x.detach())
                loss.backward()  # Backward pass.
                optimizer.step()  # Update model parameters.
                # tepoch.set_postfix(loss=loss.item(), chamfer=chamferloss.item(), ce=celoss.item(), mse = mseloss.item())
                tepoch.set_postfix(loss=loss.item(), chamfer=reconstruction_loss.item(), mse = mseloss.item())

                epoch_loss += loss.item()
                epoch_rec_loss += reconstruction_loss.item()

                mse_loss += mseloss.item()

        if epoch % 5 == 0 and val_loader is not None:
            val_loss = 0
            val_rec_loss = 0
            val_kl_loss = 0
            val_mseloss = 0
            model.eval()
            for val_idx, (valdata, _) in enumerate(val_loader):
                val_x = valdata.float().to(device)
                if type(model).__name__ == "PointVAE":
                    reconstructed, mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z = model(val_x)           
                    KLD_element_x = torch.pow(mu_x, 2) + torch.pow(
                        sigma_x, 2) - torch.log(1e-8 + torch.pow(sigma_x, 2)) - 1.0
                    KLD_element_y = torch.pow(mu_y, 2) + torch.pow(
                        sigma_y, 2) - torch.log(1e-8 + torch.pow(sigma_y, 2)) - 1.0
                    KLD_element_z = torch.pow(mu_z, 2) + torch.pow(
                        sigma_z, 2) - torch.log(1e-8 + torch.pow(sigma_z, 2)) - 1.0
                    loss_KL = kl_coeff * (torch.sum(KLD_element_x) + torch.sum(KLD_element_y) + torch.sum(KLD_element_z))
                    val_kl_loss += loss_KL
                elif type(model).__name__ == "PointAutoEncoder":
                    _, reconstructed = model(val_x)
                reconstruction_loss, _ = criterion(reconstructed, val_x)
                if type(model).__name__ == "PointVAE":
                    val_loss += loss_KL + reconstruction_loss
                elif type(model).__name__ == "PointAutoEncoder":
                    val_loss += reconstruction_loss
                if val_idx == rand_val_idx:
                    wandb.log({"epoch":epoch, "Val Ground Truth": wandb.Object3D({"type": "lidar/beta","points":val_x[rand_val_sub_idx, :, :]}), "Val Reconstructed": wandb.Object3D({"type": "lidar/beta","points":reconstructed[rand_val_sub_idx, :, :]})},commit=False)
                    val_plotly_steps.append({"epoch":epoch, "x": x[rand_val_sub_idx, :, 0].detach().cpu().numpy(), "y": x[rand_val_sub_idx, :, 1].detach().cpu().numpy(), "z": x[rand_val_sub_idx, :, 2].detach().cpu().numpy(), "type":  "Ground Truth"})
                    val_plotly_steps.append({"epoch":epoch, "x": reconstructed[rand_val_sub_idx, :, 0].detach().cpu().numpy(), "y": reconstructed[rand_val_sub_idx, :, 1].detach().cpu().numpy(), "z": reconstructed[rand_val_sub_idx, :, 2].detach().cpu().numpy(), "type":  "Ground Truth"})
                val_rec_loss += reconstruction_loss
                val_mseloss += mse(reconstructed.detach(), val_x.detach()).item()
            if intermediate_save_path is not None:
                torch.save(model.state_dict(), intermediate_save_path[2])
        base_dict = {"epoch": epoch, "train_loss": epoch_loss/len(tepoch), "train_rec_loss": epoch_rec_loss/len(tepoch), "train_kl_loss": epoch_kl_loss/len(tepoch), "train_mseloss":mseloss/len(train_loader),
                "val_loss":val_loss/len(val_loader),  "val_rec_loss": val_rec_loss/len(val_loader), "val_mseloss":val_mseloss/len(val_loader)}
        if type(model).__name__ == "PointVAE":
            base_dict = {"epoch": epoch, "train_loss": epoch_loss/len(tepoch), "train_rec_loss": epoch_rec_loss/len(tepoch), "train_kl_loss": epoch_kl_loss/len(tepoch), "train_mseloss":mseloss/len(train_loader),
                "val_loss":val_loss/len(val_loader),  "val_rec_loss": val_rec_loss/len(val_loader), "val_kl_loss": val_kl_loss/len(val_loader), "val_mseloss":val_mseloss/len(val_loader)}
        wandb.log(base_dict)
        
        if epoch == 100 and intermediate_save_path is not None:
            torch.save(model.state_dict(), intermediate_save_path[0])
        
        if epoch == 500 and intermediate_save_path is not None:
            torch.save(model.state_dict(), intermediate_save_path[1])
    layout = go.Layout(scene = dict(xaxis = dict(nticks=20, range=[-1,1],), yaxis = dict(nticks=20, range=[-1,1],), zaxis = dict(nticks=10, range=[-1,1],),aspectmode="cube"),
                  title="Reconstructed vs Ground Truth",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])],
        sliders=[dict(
            active=10,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            visible=True
        )]
        )
    fig = go.Figure(
    data=plotly_steps[0]["data"],
    layout = layout, 
    frames = plotly_steps[1:]
)
    # fig = px.scatter_3d(plotly_steps, x="x", y="y", z="z", color="type",animation_frame="epoch")
    fig.write_html("./test.html")
    return model