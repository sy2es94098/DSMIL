class Simsiam_weight(Pretrain_model):
    def __init__():
        super().__init__(args)

    def get_pretrain_weight_path(self):
        return  self.weight_path

    def load_weight(self):
        state_dict_weights = torch.load(weight_path)
            if 'tar' in weight_path:
                state_dict_weights = state_dict_weights['state_dict']
            for i in range(25):
                state_dict_weights.popitem()
        return state_dict_weights