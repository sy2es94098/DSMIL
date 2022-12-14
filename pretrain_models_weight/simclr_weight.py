class Simclr_weight(Pretrain_model):
    def __init__():
        super().__init__(args)

    def get_pretrain_weight_path(self):
        if self.weight_path is not None:
            self.weight_path = os.path.join('simclr', 'runs', self.weight_path, 'checkpoints', 'model.pth')
        else:
            self.weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
        
        return self.weight_path

    def load_weight(self):
        state_dict_weights = torch.load(self.weight_path)
        state_dict_weights = state_dict_weights['state_dict']
        for i in range(25):
            state_dict_weights.popitem()
        return state_dict_weights