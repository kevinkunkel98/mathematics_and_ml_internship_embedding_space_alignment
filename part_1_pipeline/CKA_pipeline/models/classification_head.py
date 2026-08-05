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
        backbone_inputs = {k: v for k, v in inputs.items() if k != 'labels'}

        if self.m_type == "vision":
            outputs = self.backbone(**backbone_inputs)
            hidden = outputs.last_hidden_state[:, 0, :]

        elif self.m_type == "language":
            outputs = self.backbone(**backbone_inputs, output_hidden_states=True)
            last_layer = outputs.hidden_states[-1]
            attn = backbone_inputs["attention_mask"]
            last_idx = attn.sum(dim=1) - 1
            hidden = last_layer[torch.arange(last_layer.size(0)), last_idx]
        else:
            raise ValueError("Selected model type is not known.")

        hidden = hidden.to(self.classifier.weight.dtype)
        logits = self.classifier(hidden)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits, hidden
        return logits