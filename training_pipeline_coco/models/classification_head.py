import torch.nn as nn 

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
            hidden = outputs.last_hidden_state[:, -1, :]  # [batch, hidden_dim]
        else: 
            raise ValueError("Selected model type is not known. Change in run_config.yaml")

        logits = self.classifier(hidden)  # [batch, 80]

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits

        return logits