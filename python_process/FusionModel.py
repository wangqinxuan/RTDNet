import torch
import torch.nn as nn

class PlusFusion(nn.Module):
    def __init__(self, in_channels):
        super(PlusFusion, self).__init__()
   
    def forward(self, x1,x2):
        out=x1+x2      
        return out

class ConcateFusion(nn.Module):
    def __init__(self, in_channels):
        super(ConcateFusion, self).__init__()
        self.conv1 = nn.Conv3d(in_channels*2,in_channels,
                            kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
    def forward(self, x1,x2): 
        x=torch.cat((x1, x2),1) 
        out=self.conv1(x)
        return out

class WeightedPlusFusion(nn.Module):
    def __init__(self, in_channels):
        super(WeightedPlusFusion, self).__init__()
        self.conv1 = nn.Conv3d(in_channels*2,1,
                            kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        self.avgpool=nn.AdaptiveAvgPool3d((1,1,2))
        self.sigmod=nn.Sigmoid()
        
    def forward(self, x1,x2):
        x=torch.cat((x1, x2),1) 
        x=self.conv1(x)
        w=self.avgpool(x).view(-1,2)
        w=self.sigmod(w)
        w1=torch.mean(w[:,0])
        w2=torch.mean(w[:,1])
        out=w1*x1+w2*x2
        
        return out

class AttenFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttenFusion, self).__init__()
        self.conv1 = nn.Conv3d(in_channels*2,in_channels,
                            kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        self.conv2 = nn.Conv3d(in_channels,in_channels,
                            kernel_size=(3, 5, 5), stride=(1, 1, 1), padding='same')
        self.sigmod=nn.Sigmoid()
        
    def atten(self,x):
        x_in=x
        x=self.conv2(x)
        x_mask=self.sigmod(x)
        x_atten=x_in.mul(x_mask)
        return x_atten
    
    def forward(self, x1,x2):
        x=torch.cat((x1, x2),1)
        x=self.conv1(x)
        out=self.atten(x)
        return out

    
    
