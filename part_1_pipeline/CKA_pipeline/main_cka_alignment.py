import torch 
import yaml 
from data.dataloader import load_data
from models.load_model import load_model
from training.training_cka_alignment import fit_alignment 
from training.output_manager import OutputManager
import os 

def main(): 
    ### Lade Configs
    with open("configs/run_config.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)
    
    device = torch.device(run_cfg["running_params"]["device"] if torch.cuda.is_available() else "cpu")
    
    with open("configs/vision.yaml", "r") as f:
        vision_cfg = yaml.safe_load(f)   
    with open("configs/language.yaml", "r") as f:
        language_cfg = yaml.safe_load(f)       
     
    
    # ==========================================
    # 1. VISION MODEL LADEN (TEACHER - FROZEN)
    # ==========================================
    run_cfg['model']['m_type'] = "vision"
    vision_model, vision_processor = load_model(run_cfg, vision_cfg, run_cfg['data']['n_classes'])

    # Trainierten Phase-1-Vision-Checkpoint laden (analog zum Language-Modell).
    vision_ckpt_path = vision_cfg.get("path")
    if vision_ckpt_path:
        checkpoint = torch.load(vision_ckpt_path, map_location=device)
        # save_checkpoint() speichert den state_dict unter dem Key "model";
        # falls du nur den nackten state_dict gespeichert hast, greift der else-Zweig.
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        missing, unexpected = vision_model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Trainiertes Vision-Modell geladen aus: {vision_ckpt_path}")
        if missing:
            print(f"  -> fehlende Keys ({len(missing)}):", missing)
        if unexpected:
            print(f"  -> unerwartete Keys ({len(unexpected)}):", unexpected)
    else:
        print("[INFO] Kein 'path' in vision.yaml gesetzt -> frisches vortrainiertes Modell.")

    vision_model = vision_model.to(device)

    # Einfrieren! Das Vision-Modell darf nicht lernen.
    for param in vision_model.parameters():
        param.requires_grad = False
    vision_model.eval()

    # ==========================================
    # 2. LANGUAGE MODEL LADEN (STUDENT - TRAINABLE)
    # ==========================================
    run_cfg['model']['m_type'] = "language"
    language_model, language_processor = load_model(run_cfg, language_cfg, run_cfg['data']['n_classes'])
    
    ckpt_path = language_cfg["path"]
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        # save_checkpoint() speichert den state_dict unter dem Key "model";
        # falls du irgendwo nur den nackten state_dict gespeichert hast, greift der else-Zweig.
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        missing, unexpected = language_model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Trainiertes Language-Modell geladen aus: {ckpt_path}")
        if missing:
            print(f"  -> fehlende Keys ({len(missing)}):", missing)
        if unexpected:
            print(f"  -> unerwartete Keys ({len(unexpected)}):", unexpected)
    else:
        print("[INFO] Kein 'path' in language.yaml gesetzt -> frisches vortrainiertes Modell.")
    
    
    language_model = language_model.to(device)

    # ==========================================
    # 3. DATEN & OUTPUT MANAGER
    # ==========================================
    run_cfg['model']['m_type'] = "alignment"
    train_ldr, test_ldr = load_data(
        run_cfg=run_cfg,
        processor=None,
        vision_processor=vision_processor, 
        language_processor=language_processor
    )

    exp_name = run_cfg['data']['experiment_name']
    root_dir = os.path.join("./models", "alignment_phase2")
    output_manager = OutputManager(experiment_name=exp_name, root_dir=root_dir, config=run_cfg)
    
    # ==========================================
    # 4. DAS TRAINING STARTEN
    # ==========================================
    print("Starte Phase 2: Geometrie-Alignment mit CKA...")
    
    language_model, train_losses, val_losses = fit_alignment(
        vision_model=vision_model,
        language_model=language_model,
        train_loader=train_ldr, 
        test_loader=test_ldr, 
        run_cfg=run_cfg,
        device=device, 
        output_manager=output_manager,
        alpha=run_cfg['running_params']['alpha'], 
        beta=run_cfg['running_params']['beta'] # Teste hier Werte zwischen 0.05 und 0.5
    )
    
    print("Phase 2 abgeschlossen!")

if __name__ == "__main__":
    main()