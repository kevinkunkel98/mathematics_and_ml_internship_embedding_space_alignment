import os 
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch 

class COCODs(Dataset): 
    def __init__(self, run_cfg,processor, DATA_TYPE ='train2017', task="classification"):
        self.instances_json = os.path.join(run_cfg['data']['path'], 'annotations', f'instances_{DATA_TYPE}.json')
        self.captions_json = os.path.join(run_cfg['data']['path'], 'annotations', f'captions_{DATA_TYPE}.json')
        self.image_dir = os.path.join(run_cfg['data']['path'], DATA_TYPE)
        
        self.processor = processor
        self.task = task
        self.model_type = run_cfg['model']['m_type'] 
        
        ### loads instances of dataset (ids)
        self.coco_inst = COCO(self.instances_json)
        self.coco_caps = COCO(self.captions_json)
        self.img_ids   = self.coco_inst.getImgIds()

        # Fixed mapping: category_id → index in label vector
        self.cat_ids        = sorted(self.coco_inst.getCatIds())  # all 80 COCO categories
        self.cat_id_to_idx  = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.num_classes    = len(self.cat_ids)  # should be 80
        
    def _get_label_vector(self, img_id):
        """Build binary vector encoding which labels are correct for this image"""
        ann_ids = self.coco_inst.getAnnIds(imgIds=img_id)
        anns    = self.coco_inst.loadAnns(ann_ids)

        label_vector = torch.zeros(self.num_classes)
        for ann in anns:
            idx = self.cat_id_to_idx[ann['category_id']]
            label_vector[idx] = 1.0

        return label_vector  # e.g. [0, 1, 0, 0, 1, 0, ..., 1]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id       = self.img_ids[idx]
        label_vector = self._get_label_vector(img_id)  # same for both model types

        if self.model_type == "vision":
            """loads images as inputs and label vectors as output for model"""
            img_info = self.coco_inst.loadImgs(img_id)[0]
            image    = Image.open(os.path.join(self.image_dir, img_info['file_name'])).convert("RGB")
            inputs   = self.processor(images=image, return_tensors="pt")
            ### adds a batch dimension 
            inputs   = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['labels'] = label_vector
            return inputs

        elif self.model_type == "causal_lm":
            """loads captions as inputs and label vectors as output for model"""
            cap_ann_ids = self.coco_caps.getAnnIds(imgIds=img_id)
            ann     = self.coco_caps.loadAnns(cap_ann_ids)[0]
            caption = ann.get('caption', '')  # .get() bypasses the TypedDict restriction
            inputs      = self.processor(caption, return_tensors="pt", truncation=True, max_length=128)
            inputs      = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['labels'] = label_vector
            return inputs


def load_data(run_cfg, processor):
    
    # 1. Create train and test dataset
    train_ds = COCODs(run_cfg=run_cfg,
                        DATA_TYPE='train2017', 
                        processor=processor)
    
    test_ds = COCODs(run_cfg=run_cfg,
                        DATA_TYPE='val2017', 
                        processor=processor)
    
    # 2. Create train and test dataloader 
    
    train_loader = DataLoader(train_ds, 
                              batch_size=run_cfg['running_params']['batch_size'], 
                                shuffle=run_cfg['running_params']['shuffle'],
                                num_workers=run_cfg['running_params']['n_workers'])

    test_loader = DataLoader(test_ds, 
                              batch_size=run_cfg['running_params']['batch_size'], 
                                shuffle=False,
                                num_workers=run_cfg['running_params']['n_workers'])
    
    return train_loader, test_loader
    
    
    
    
    