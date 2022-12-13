import torch
import torchvision.models as models
import torch.nn as nn


state_dict = torch.load("simclr/runs/Sep16_07-57-17_server/checkpoints/model.pth")
#state_dict = torch.load("simsiam/checkpoint_0003.pth.tar")
#state_dict = state_dict['state_dict']
'''
for k in list(state_dict.keys()):
    #print(k)
    # retain only encoder up to before the embedding layer
    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
        # remove prefix
        #print(k[len("module.encoder."):])
        state_dict[k[len("module.encoder."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]
'''
#norm=nn.InstanceNorm2d
#pretrain = False
#state_dict = models.resnet50().state_dict()

#model = models.__dict__['resnet50'](zero_init_residual=True, pretrained=False)
#state_dict = model.state_dict()

#for i in range(4):
#    state_dict.popitem()
for i in state_dict:
    print(i)

