import abc

class Pretrain_model(metaclass=abc.ABCMeta):
    
    def __init__(self,args):
        self.weight_path = args.weights

    def init_model(self):
        self.load_weight()
        self.load_pretrain_backbone()

    def get_pretrain_weight_path(self):
        return self.weight_path

    def show_model_arch(self):
        for k,v in self.state_dict_weights.items():
            print(k)

    def show_backbone_arch(self):
        for k,v in self.backbone_state_dict.items():
            print(k)

    def get_model(self):
        return self.state_dict_weights

    def get_backbone(self):
        return self.backbone_state_dict

    @abc.abstractmethod
    def load_weight(self):
        return NotImplemented

    @abc.abstractmethod
    def load_pretrain_backbone(self):
        return NotImplemented

