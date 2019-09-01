import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn


def softmax_mse_loss(out1, out2):
        assert out1.size() == out2.size()
        m = nn.Softmax(dim=1)
        out1 = m(out1).contiguous().view(-1)
        out2 = m(out2).contiguous().view(-1)
        diff = torch.sum((out1 - out2)**2)
#        print(out1.data.nelement())
        return diff / out1.data.nelement()

#        return F.mse_loss(out1,out2)


def softmax_kl_loss(out1, out2):
    assert out1.size() == out2.size()
    m = nn.Softmax(dim=1)
    out1 = m(out1).contiguous().view(-1)
    out2 = m(out2).contiguous().view(-1)
    return F.kl_div(out1,out2)/out1.data.nelement()

def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)
