from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
import os
import yaml  
import torch 
from peft import LoraConfig, get_peft_model, TaskType ### library for efficiently addapting pretrained models 
from models.classification_head import MultiLabelModel

### load model (either languguage or vision)
def load_model(run_cfg, model_cfg, n_classes): 
        if run_cfg['model']['m_type'] == "vision": 
                processor = AutoImageProcessor.from_pretrained(model_cfg['model_id'])
                model = AutoModel.from_pretrained(model_cfg['model_id'])     
                
                ### LORA adds weights to the pre-trained model, which are finetuned for task 
                lora_config = LoraConfig(
                        task_type=None,
                        r=model_cfg['lora']['lora_r'], ### rank of update metrices (lower adds fewer trainable parameters)
                        lora_alpha=model_cfg['lora']['lora_alpha'], ### Lora scaling factor ??? 
                        target_modules=model_cfg['lora']['target_modules'],  # where should be parameters added 
                        bias="none"
                        )  
                
        elif  run_cfg['model']['m_type'] == "language":        
                processor = AutoTokenizer.from_pretrained(
                        model_cfg['model_id'],
                        token=model_cfg['token'])
                
                if processor.pad_token is None:
                        processor.pad_token = processor.eos_token
                        
                model = AutoModelForCausalLM.from_pretrained(
                        model_cfg['model_id'],
                        token=model_cfg['token'],
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                        )  
                
                
                lora_config = LoraConfig(
                        task_type=None,
                        r=model_cfg['lora']['lora_r'],
                        lora_alpha=model_cfg['lora']['lora_alpha'],
                        target_modules=model_cfg['lora']['target_modules'],  # e.g. ["q_proj", "v_proj"]
                        bias="none"
                        )
        else: 
                raise ValueError("Selected model type is not known. Change in run_config.yaml")
        
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()     
        
        model = MultiLabelModel(model, model_cfg, n_classes)
        
        return model, processor            
        

