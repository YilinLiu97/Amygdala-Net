import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from Dataset import Crop
import torch.optim as optim
import math
from AdaDropout_dynamic import AdaDropout

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, do_act, if_drop, wrs_ratio_fc, drop_rate_fc, test_state):
       super(ConvBlock, self).__init__()
       self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

       self.bn = nn.BatchNorm3d(out_channels)
       self.do_act = do_act
       self.if_drop = if_drop
       if self.do_act:
         self.act = nn.ReLU()

       if self.if_drop:
          self.dropout = nn.Dropout3d(0) #AdaDropout(out_channels, test_state)   #SpatialSELayer(out_channels)  #AdaDropout(out_channels, test_state) #AdaDropout_RNG_Only(out_channels, test_state)   #AdaDropout(out_channels, test_state)  

    def forward(self, input):
       out = self.bn(self.conv(input))

       if self.do_act:
          out = self.act(out)
       if self.if_drop:
          out = self.dropout(out)

       return out


class makePath(nn.Module):

    def __init__(self, channels_list, dilation_list, kernel_size, num_convs, if_drop, wrs_ratio_fc, drop_rate_fc, test_state):
       super(makePath, self).__init__()
       layers = []
       for i in range(num_convs):
           if i != num_convs - 1:
              if dilation_list == None:
                 layers.append(ConvBlock(channels_list[i],channels_list[i+1],kernel_size,1,True,if_drop, wrs_ratio_fc, drop_rate_fc, test_state))
              else:
                 layers.append(ConvBlock(channels_list[i],channels_list[i+1],kernel_size,dilation_list[i],True,if_drop, wrs_ratio_fc, drop_rate_fc, test_state))
           else:
              pass

       self.path = nn.Sequential(*layers)

    def forward(self, input):

        output = self.path(input)

        return output


class AmygNet3D(nn.Module):

    def __init__(self, num_classes, wrs_ratio, drop_rate, wrs_ratio_fc, drop_rate_fc, test_state=False):
        super(AmygNet3D,self).__init__()

        self.test_state = test_state

        self.drop_rate = drop_rate
        self.wrs_ratio = wrs_ratio

        self.drop_rate_fc = drop_rate_fc
        self.wrs_ratio_fc = wrs_ratio_fc

        #normal path
        self.normal_path_chns = [1,30,30,40,40,40,40,50,50,50,50]
        self.normal_dilation_list = None

        self.normal_path = makePath(self.normal_path_chns, self.normal_dilation_list, 3, len(self.normal_path_chns), False, wrs_ratio_fc, drop_rate_fc, test_state)

        #dilated path
        self.dilated_path_chns = [1,30,30,40,40,40,40,50,50,50]
        self.dilated_dilation_list = [1,2,4,2,8,2,4,2,1]

        self.dilated_path = makePath(self.dilated_path_chns, self.dilated_dilation_list, 3, len(self.dilated_path_chns), False, wrs_ratio_fc, drop_rate_fc, test_state)


        #FC layers
        self.FC_chns = [100,150,150]
        self.FC_dilation_list = [1,1]

        self.FC = makePath(self.FC_chns, self.FC_dilation_list, 1, len(self.FC_chns), True, wrs_ratio_fc, drop_rate_fc, test_state) # Only drop units in the fully connected layers

        #Classification layer
        self.classification = ConvBlock(150,num_classes,1,1,False, False, wrs_ratio, drop_rate, test_state) # No activation

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
               nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm3d):
               nn.init.constant_(m.bias.data, 0.0)
               nn.init.normal_(m.weight.data, 1.0, 0.02)

    def forward(self,x,args, triple=False):

        cropped_x = Crop(x,[27,27,27])

        #normal_path
        x_normal = self.normal_path(cropped_x)

        #dilated_path
        x_dilated = self.dilated_path(x)

        x_merge = torch.cat((x_normal,x_dilated),1)

        if triple:
           att = AdaDropout(100, self.wrs_ratio, self.test_state)
           x_FC = self.FC(att(x_merge))
        else:
           x_FC = self.FC(x_merge)

        output = self.classification(x_FC)
        return output
