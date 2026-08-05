import torch.nn as nn 
import torch 

class MultiLabelModel(nn.Module): 
    def __init__(self, backbone_model, model_cfg, n_classes):
        super().__init__()
        
        self.backbone = backbone_model
        self.m_type = model_cfg['m_type']
        self.n_classes = n_classes
        
        self.hidden_dim = model_cfg['hidden_dim']
        
        self.classifier = nn.Linear(self.hidden_dim, self.n_classes)
        self.loss_fn    = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, labels=None):
        # Extract everything except labels before passing to backbone
        backbone_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
                
        outputs = self.backbone(**backbone_inputs)

        if self.m_type == "vision":
            # CLS token is the first token — summarizes the whole image
            hidden = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]

        elif self.m_type == "language":
            # Last token summarizes the sequence for causal LMs
            outputs = self.backbone(**backbone_inputs, output_hidden_states=True)
            last_layer = outputs.hidden_states[-1]               # [B, seq, hidden]

            # last *non-pad* token (right padding -> can't just use [:, -1, :])
            attn = backbone_inputs["attention_mask"]             # [B, seq]
            last_idx = attn.sum(dim=1) - 1                       # [B]
            hidden = last_layer[torch.arange(last_layer.size(0)), last_idx]  # [B, hidden]
        else: 
            raise ValueError("Selected model type is not known. Change in run_config.yaml")
        
        ### because the ouput of the language model can have dtype BFloat16
        hidden = hidden.to(self.classifier.weight.dtype) 
        logits = self.classifier(hidden)  # [batch, 80]

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits

        return logits