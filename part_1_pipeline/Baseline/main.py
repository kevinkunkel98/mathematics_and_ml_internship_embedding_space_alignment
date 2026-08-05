import torch 
import yaml 
from data.dataloader import load_data
from models.load_model import load_model
from training.training import fit 
from training.output_manager import OutputManager
import os 

def main(): 
    
    ### load running params 
    with open("configs/run_config.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)

    requested_device = run_cfg["running_params"]["device"]
    if isinstance(requested_device, str) and requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested in config but no CUDA device is available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    
    
    ### load model params 
    if run_cfg['model']['m_type'] == "vision": 
        with open("configs/vision.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)   
    elif  run_cfg['model']['m_type'] == "language": 
        with open("configs/language.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)       
    else: 
        raise ValueError("Selected model type is not known. Change in run_config.yaml") 
     
    ### load model 
    model, processor = load_model(run_cfg=run_cfg,model_cfg=model_cfg,n_classes=run_cfg['data']['n_classes'])
    
    if run_cfg['model']['m_type'] == "vision":
        model = model.to(device)
    elif run_cfg['model']['m_type'] == "language":
        model.classifier.to(device)
    ### load dataset 
    train_ldr, test_ldr = load_data(run_cfg=run_cfg, processor=processor)
    
    ### create output manager
    exp_name =  run_cfg['data']['experiment_name']
    root_dir = os.path.join("./models", run_cfg['model']['m_type'])
    
    output_manager = OutputManager(experiment_name=exp_name,
                                   root_dir=root_dir,
                                   config={**run_cfg, **model_cfg})
    
    ### fit model 
    model, train_losses, val_losses  = fit(model=model,
                                            train_loader=train_ldr, 
                                            test_loader=test_ldr, 
                                            run_cfg=run_cfg,
                                            device = device, 
                                            output_manager = output_manager)
    


if __name__ == "__main__":
    main()
