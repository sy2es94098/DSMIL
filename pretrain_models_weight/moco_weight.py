from pretrain_models_weight.pretrain_model import Pretrain_model
import torch

class Moco_weight(Pretrain_model):
    def __init__(self, args):
        super().__init__(args)
        self.init_model()

    def load_weight(self):
        self.state_dict_weights = torch.load(self.weight_path)
        self.state_dict_weights = self.state_dict_weights['state_dict']
        return self.state_dict_weights

    def load_pretrain_backbone(self):
        state_dict_weights = self.state_dict_weights.copy()
        for k in list(state_dict_weights.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict_weights[k[len("module.encoder_q."):]] = state_dict_weights[k]
            # delete renamed or unused k
            del state_dict_weights[k]

        for i in range(0):
            state_dict_weights.popitem()

        self.backbone_state_dict = state_dict_weights

        return self.backbone_state_dict



