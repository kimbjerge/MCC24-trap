'''Deep Hierarchical Classifier using resnet50 with cbam as the base.
'''

import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class EfficientNetBx(nn.Module):
    '''EfficientNet Architecture with pretrained weights
    '''

    def __init__(self, use_cbam=True, image_depth=3, num_classes=20, eff_name='b4'):
        '''Params init and build arch.
        '''
        super(EfficientNetBx, self).__init__()

        self.expansion = 4
        self.out_channels = 320 # 320*4 = 1280, B0
        if eff_name == 'b3':
       	    self.out_channels = 384 # 384*4 = 1536, B3
        if eff_name == 'b4':
            self.out_channels = 448 # 448*4 = 1792, B4
        
        self.model_ft = EfficientNet.from_pretrained(f'efficientnet-{eff_name}')
        
        # optionally freeze parameters
        #for p in self.model_ft.parameters():
        #  p.requires_grad = False
        
        # fintune layer4
        #for p in self.model_ft.layer4.parameters():
        #  p.requires_grad = True
        
        # overwrite the 'fc' layer
        print("In features", self.model_ft._fc.in_features) # 1280
        self.model_ft._fc = nn.Identity() # Do nothing just pass input to output
        
        # At least one layer
        #self.linear_lvl1 = nn.Linear(512*self.expansion, num_classes[0])
        #self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(self.out_channels*self.expansion, self.out_channels)
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        '''Forward propagation of pretrained EfficientNet.
        '''
        x = self.model_ft(x)
        
        x = self.drop(x) # Dropout to add regularization

        level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        #level_1 = nn.Softmax(level_1)
                
        return level_1
    