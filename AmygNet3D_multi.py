import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from Dataset import Crop
import torch.optim as optim
import math
from TFD import TFD

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, do_act=True, if_drop=False, test_state=False):
       super(ConvBlock, self).__init__()

       self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

       self.bn = nn.BatchNorm3d(out_channels)
       self.do_act = do_act
       self.if_drop = if_drop
       if self.do_act:
         self.act = nn.ReLU()

       if self.if_drop:
          self.dropout = nn.Dropout3d(0)  #TFD(out_channels, test_state)

    def forward(self, input):
       out = self.bn(self.conv(input))

       if self.do_act:
          out = self.act(out)
       if self.if_drop:
          out = self.dropout(out)

       return out

class HybridConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, do_act=True):
       super(HybridConvBlock, self).__init__()

       self.dilated_conv = ConvBlock(in_channels, out_channels, 3, dilation_rate, do_act)
       self.normal_conv = ConvBlock(in_channels, out_channels, 3, 1, do_act)

    def forward(self, input):
       x_dilated = self.dilated_conv(input)
       x_normal = self.normal_conv(input)
       _,_,a,b,c = x_dilated.size()
       x_normal = Crop(x_normal,[a,b,c])
       x_normal = x_normal
       out = x_dilated + x_normal
       return out

class GlobalRefine(nn.Module):
    def __init__(self, out_channels_hg, out_channels_low):
       super(GlobalRefine, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool3d(1)
       self.conv1x1x1 = ConvBlock(out_channels_hg, out_channels_low, 1, 1, True)
       self.reshape = ConvBlock(out_channels_low, out_channels_hg, 1, 1, False) # No activation
       self.sigmoid = nn.Sigmoid()
       
    def forward(self, conv_high, conv_low):
        _,_,A,B,C = conv_high.size()
        global_att = self.avg_pool(conv_high).to(dtype=torch.float).cuda()
        global_att = self.sigmoid(self.conv1x1x1(global_att))
        reweighted_conv_low = conv_low*global_att
        reweighted_conv_low = Crop(reweighted_conv_low, [A,B,C])
        reweighted_conv_low = self.reshape(reweighted_conv_low)
        return conv_high + reweighted_conv_low

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, dilation_list, do_act=True, if_drop=False, test_state=False):
       super(ResBlock, self).__init__()

       self.conv_relu = HybridConvBlock(in_channels, out_channels_1, dilation_list[0], True) # Do activation
       self.conv = HybridConvBlock(out_channels_1, out_channels_2, dilation_list[1], False) # No activation
       self.reshape = ConvBlock(in_channels, out_channels_2, 1, 1, False) # No activation
     

       self.relu = nn.ReLU()

    def forward(self, x):
       identity = x
       out = self.conv_relu(x)
       out = self.conv(out)
       _,_,a,b,c = out.size()
       identity = Crop(identity,[a,b,c])
       identity = self.reshape(identity)
       out += identity
       out = self.relu(out)
       return out


class AmygNet3D(nn.Module):

    def __init__(self, num_classes, wrs_ratio, drop_rate, wrs_ratio_fc, drop_rate_fc, test_state=False):
        super(AmygNet3D,self).__init__()

        self.test_state = test_state

        self.firstConv = HybridConvBlock(1,30,1) # rate=2, do_act=True

        #dilated path
        self.g1 = ResBlock(30,30,40,[2,4])
        self.g2 = ResBlock(40,40,40,[2,8])
        self.g1_refine_g2 = GlobalRefine(40,40)
        self.g3 = ResBlock(40,40,50,[2,4])
        self.g2_refine_g3 = GlobalRefine(50,40)
        self.g4 = ResBlock(50,50,50,[2,1])
        self.g3_refine_g4 = GlobalRefine(50,50)


        #FC layers
        self.FC_1 = ConvBlock(50,150,1,1,True,True,self.test_state)
        self.FC_2 = ConvBlock(150,150,1,1,True,True,self.test_state)

        #Classification layer
        self.classification = ConvBlock(150,num_classes,1,1,False, False) # No activation, AdaDropout

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
               nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm3d):
               nn.init.constant_(m.bias.data, 0.0)
               nn.init.normal_(m.weight.data, 1.0, 0.02)

    def forward(self,x,args):
        out = self.firstConv(x)
        g1_out = self.g1(out)
        g2_out = self.g2(g1_out)
        g3_out = self.g3(g2_out)
        g4_out = self.g4(g3_out)
        
        refined_g2 = self.g1_refine_g2(g2_out, g1_out) 
        refined_g3 = self.g2_refine_g3(g3_out, refined_g2)
        refined_g4 = self.g3_refine_g4(g4_out, refined_g3) 
        

        if args.triple:
           att = AdaDropout(100, self.wrs_ratio, self.test_state)
           out = self.FC_1(att(out))
        else:
           out = self.FC_1(refined_g4)
           out = self.FC_2(out)

        out = self.classification(out)
        return out
