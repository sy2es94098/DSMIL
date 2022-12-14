class Simsiam_weight(Pretrain_model):
    def __init__():
        super().__init__(args)

    def get_pretrain_weight_path(self):
        return  self.weight_path

    def load_weight(self):
        state_dict_weights = torch.load(self.weight_path)
        for i in range(4):
            state_dict_weights.popitem()
            
        return state_dict_weights