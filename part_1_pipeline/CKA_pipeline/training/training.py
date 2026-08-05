import torch 
from torch.utils.data import Dataset, DataLoader
from training.early_stopping import EarlyStopping
#from checkpoints import CheckpointManager 
from torch.optim import AdamW
import pandas as pd 

def test(model, ldr, device): 
    loss_t = 0.0 
    model.eval()
    with torch.no_grad():
        for batch in ldr: 
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch, labels=batch['labels'])
            loss_t += loss.item()
            
    return loss_t/ len(ldr) 
    
def train(model, ldr, optimizer: AdamW, device):
    loss_t = 0.0 
    model.train() 
    for batch in ldr: 
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(batch, labels=batch['labels'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_t += loss.item()
        
    return loss_t / len(ldr)
        

def fit(model, train_loader, test_loader, run_cfg, device, output_manager):
    train_losses = []
    val_losses = []
    
    ### initialize optimizer
    optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr = run_cfg['running_params']['lr']
                    )
    
    early = EarlyStopping(run_cfg['running_params']['patience'])
    
    for epoch in range(run_cfg['running_params']['n_epochs']):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss   = test(model, test_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # Simple checkpoint — just save the best model
        if val_loss == min(val_losses): 
            output_manager.save_checkpoint(model, optimizer, epoch, val_loss)
            print(f"  → best model saved (epoch {epoch+1}, val_loss: {val_loss:.4f})")

        early(val_loss)
        if early.early_stop:
            print("Early stopping triggered.")
            break
        
    output_manager.save_dataframe(
        pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}),
        filename="losses.csv",
        subdir="metrics"
    )


    return model, train_losses, val_losses  