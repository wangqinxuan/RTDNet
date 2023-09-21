import torch
import torch.nn as nn
from FusionModel import ConcateFusion as Fusion


class CDModel(nn.Module):

    def __init__(self, nb_channels,n_class=1):
        super(CDModel, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=nb_channels[0],
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        self.conv1b = nn.Conv3d(in_channels=nb_channels[0], out_channels=nb_channels[0],
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        
        
        self.conv2a = nn.Conv3d(in_channels=nb_channels[0], out_channels=nb_channels[1],
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        self.conv2b = nn.Conv3d(in_channels=nb_channels[1], out_channels=nb_channels[1],
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        
        
        self.conv3a = nn.Conv3d(in_channels=nb_channels[1], out_channels=nb_channels[2],
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        self.conv3b = nn.Conv3d(in_channels=nb_channels[2], out_channels=nb_channels[2],
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        
        self.conv_atten1 = nn.Conv3d(nb_channels[2], nb_channels[2],(3, 5, 5),(1, 1, 1),padding='same')
        self.conv_atten2 = nn.Conv3d(nb_channels[0], nb_channels[0],(3, 5, 5),(1, 1, 1),padding='same')
        
        self.bn1a = nn.BatchNorm3d(num_features=nb_channels[0])
        self.bn1b = nn.BatchNorm3d(num_features=nb_channels[0])
        self.bn2a = nn.BatchNorm3d(num_features=nb_channels[1])
        self.bn2b = nn.BatchNorm3d(num_features=nb_channels[1])
        self.bn3a = nn.BatchNorm3d(num_features=nb_channels[2])
        self.bn3b = nn.BatchNorm3d(num_features=nb_channels[2])
        
        self.maxpool1=nn.MaxPool3d((1,2,2))
        self.maxpool2=nn.MaxPool3d((1,2,2))
        self.maxpool3=nn.MaxPool3d((1,2,2))
        
        self.avgpool=nn.AdaptiveMaxPool3d((4,147,334))
        
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        
        self.fusion1=Fusion(nb_channels[0])
        self.fusion2=Fusion(nb_channels[1])
        self.fusion3=Fusion(nb_channels[2])
        
        self.convt1 = nn.ConvTranspose3d(in_channels=nb_channels[2], out_channels=nb_channels[1],
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=nb_channels[1], out_channels=nb_channels[0],
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=nb_channels[0], out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.maxpool1(x))
        
        x = self.relu(self.bn2a(self.conv2a(x)))  
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.relu(self.bn2b(self.conv2b(x))) 
        x = self.relu(self.maxpool2(x))
        
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.relu(self.maxpool3(x))

        x = self.prelu(self.convt1(x)) 
        x = self.prelu(self.convt2(x)) 
        x = self.convt3(x)
        x = self.avgpool(x)
        return x