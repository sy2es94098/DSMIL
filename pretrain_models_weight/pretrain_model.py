import abc

class Pretrain_model(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self,args):
        self.weight_path = args.weights

    @abc.abstractmethod
    def get_pretrain_weight_path(self):
        return NotImplemented
 
    @abc.abstractmethod
    def load_weight(self):
        return NotImplemented
