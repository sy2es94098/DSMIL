from pretrain_models_weight.pretrain_model import Pretrain_model
import torch

class Simsiam_weight(Pretrain_model):
    def __init__(self, args):
        super().__init__(args)
        self.init_model()

    def load_weight(self):
        self.state_dict_weights = torch.load(self.weight_path)
        self.state_dict_weights = self.state_dict_weights['state_dict']
        return self.state_dict_weights

    def load_pretrain_backbone(self):
        state_dict_weights = self.state_dict_weights.copy()

        for i in range(25):
            state_dict_weights.popitem()

        self.backbone_state_dict = state_dict_weights

        return self.backbone_state_dict



