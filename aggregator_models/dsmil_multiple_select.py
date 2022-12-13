import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from skimage import exposure

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc =  nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc =  nn.Sequential(nn.Linear(feature_size, output_class))

        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False, n_critical=3): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU())
        else:
            self.v = nn.Identity()

        self.n_critical=n_critical
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        A_sum = torch.zeros(m_indices.shape).cuda()
        B_sum = torch.zeros(1, m_indices.shape[1], V.shape[1]).cuda()
        C_sum = torch.zeros(1, m_indices.shape[1]).cuda()

        #B_sum = torch.zeros()
        #m_feats_list = []
        #A_list = torch.tensor([])
        #B_list = torch.tensor([])
        #C_list = torch.tensor([])
        #A_sum = torch.zeros(
        for i in range(self.n_critical):
            m_feats = torch.index_select(feats, dim=0, index=m_indices[i, :])
            q_max = self.q(m_feats)
            A = torch.mm(Q, q_max.transpose(0, 1))
            A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
            B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V


            B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
            C = self.fcc(B) # 1 x C x 1
            C = C.view(1, -1)
           
            A_sum = torch.add(A_sum, A)
            B_sum = torch.add(B_sum, B)
            C_sum = torch.add(C_sum, C)

            #print('A------------------------')
            #print(A_sum)
            #print('B------------------------')
            #print(B_sum)
            #print('C------------------------')
            #print(C_sum)

            #m_feats_list.append(m_feats)
            #A_list.append(A)
            #B_list.append(B)
            #C_list.append(C)
        
        '''
        m_feats_list = np.array(m_feats_list)
        A_list = np.array(A_list)
        B_list = np.array(B_list)
        C_list = np.array(C_list)
        '''
        A = torch.div(A_sum, self.n_critical)
        B = torch.div(B_sum, self.n_critical)
        C = torch.div(C_sum, self.n_critical)
        #A = torch.mean(A_list, axis=0)
        #B = torch.mean(B_list, axis=0)
        #C = torch.mean(C_list, axis=0)
        



        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B
        
