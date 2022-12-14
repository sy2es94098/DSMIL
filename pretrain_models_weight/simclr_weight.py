from pretrain_models_weight.pretrain_model import Pretrain_model

import os
import torch

class Simclr_weight(Pretrain_model):
    def __init__(self, args):
        super().__init__(args)
        if args.weights is not None:
            self.weight_path = os.path.join('contrastive_models', 'simclr', 'runs', self.weight_path, 'checkpoints', 'model.pth')
        else:
            self.weight_path = glob.glob('contrastive_models/simclr/runs/*/checkpoints/*.pth')[-1]
        self.init_model()

    def load_weight(self):
        self.state_dict_weights = torch.load(self.weight_path)
        return self.state_dict_weights

    def load_pretrain_backbone(self):
        state_dict_weights = self.state_dict_weights.copy()
        
        for i in range(4):
            state_dict_weights.popitem()

        self.backbone_state_dict = state_dict_weights

        return self.backbone_state_dict


