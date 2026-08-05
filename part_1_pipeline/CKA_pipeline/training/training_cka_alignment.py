import torch
from torch.utils.data import Dataset, DataLoader
from training.early_stopping import EarlyStopping
#from checkpoints import CheckpointManager 
from torch.optim import AdamW
import pandas as pd
from training.losses import CKALoss

def train_alignment_epoch(vision_model, language_model, loader, optimizer, device, cka_criterion, alpha, beta):
    language_model.train()
    vision_model.eval() # Teacher bleibt immer im Eval-Modus
    
    total_loss = 0.0
    
    for batch in loader:
        # Inputs aufteilen
        labels = batch['labels'].to(device)
        vision_inputs = {k: v.to(device) for k, v in batch.items() if k in ['pixel_values']}
        language_inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        
        optimizer.zero_grad()
        
        # 1. Teacher Activations (Frozen)
        with torch.no_grad():
            _, _, vision_hidden = vision_model(vision_inputs, labels)
            
        # 2. Student Activations & Task Loss
        task_loss, _, language_hidden = language_model(language_inputs, labels)
        
        # 3. CKA Loss
        cka_loss = cka_criterion(vision_hidden, language_hidden)
        
        # 4. Kombinierter Loss & Update
        loss = (alpha * task_loss) + (beta * cka_loss)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader), task_loss, cka_loss


@torch.no_grad()
def test_alignment_epoch(vision_model, language_model, loader, device, cka_criterion, alpha, beta):
    language_model.eval()
    vision_model.eval()
    
    total_loss = 0.0
    
    for batch in loader:
        labels = batch['labels'].to(device)
        vision_inputs = {k: v.to(device) for k, v in batch.items() if k in ['pixel_values']}
        language_inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        
        _, _, vision_hidden = vision_model(vision_inputs, labels)
        task_loss, _, language_hidden = language_model(language_inputs, labels)
        
        cka_loss = cka_criterion(vision_hidden, language_hidden)
        loss = (alpha * task_loss) + (beta * cka_loss)
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


def fit_alignment(vision_model, language_model, train_loader, test_loader, run_cfg, device, output_manager, alpha=1.0, beta=0.2):
    train_losses = []
    train_task_losses = []
    train_cka_losses = []
    val_losses = []
    
    # 1. Der Optimizer bekommt NUR die Parameter des Language Models (LoRA Gewichte)
    optimizer = AdamW(
                    filter(lambda p: p.requires_grad, language_model.parameters()),
                    lr = run_cfg['running_params']['lr']
                    )
    
    early = EarlyStopping(run_cfg['running_params']['patience'])
    
    cka_criterion = CKALoss()
    print("now starting the  first epoch of",range(run_cfg['running_params']['n_epochs']) )

    for epoch in range(run_cfg['running_params']['n_epochs']):
        
        train_loss, train_task_loss, train_cka_loss  = train_alignment_epoch(vision_model, language_model, train_loader, optimizer, device, cka_criterion, alpha, beta)
        val_loss   = test_alignment_epoch(vision_model, language_model, test_loader, device, cka_criterion, alpha, beta)


        train_losses.append(train_loss)


        train_task_losses.append(train_task_loss.item() if hasattr(train_task_loss, 'item') else train_task_loss)
        train_cka_losses.append(train_cka_loss.item() if hasattr(train_cka_loss, 'item') else train_cka_loss)


        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | train loss: {train_loss:.4f} | train loss task: {train_task_loss:.4f} | train loss cka: {train_cka_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss == min(val_losses): 
            output_manager.save_checkpoint(language_model, optimizer, epoch, val_loss)
            print(f"  → best model saved (epoch {epoch+1}, val_loss: {val_loss:.4f})")

        early(val_loss)
        if early.early_stop:
            print("Early stopping triggered.")
            break
        
    output_manager.save_dataframe(
        pd.DataFrame({"train_loss": train_losses, "train_loss_taks": train_task_losses, "train_loss_cka" : train_cka_losses, "val_loss": val_losses}),
        filename="alignment_losses.csv",
        subdir="metrics"
    )

    print("\nSpeichere finales Modell nach Abschluss aller Epochen...")
    output_manager.save_checkpoint(
        model=language_model,
        optimizer=optimizer,
        epoch=epoch + 1,        # Letzte durchgelaufene Epoche
        val_loss=val_loss,      # Letzter validierter Loss
        filename="final_language_model.pt" # Verhindert das Überschreiben von 'best_model.pt'
    )
    print("Alles erfolgreich gespeichert!")

    return language_model, train_losses, val_losses, train_task_losses, train_cka_losses