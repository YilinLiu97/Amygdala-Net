import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WeightEMA(object):

    def __init__(self, model, tea_model):
       self.model = model
       self.tea_model = tea_model
       self.params = list(model.parameters())
       self.tea_params = list(tea_model.parameters())

#	for p,t_p in zip(self.params, self.tea_params):
#           t_p.data[:] = p.data[:]

    def step(self,alpha,global_step):
  #	 alpha = min(1 - 1 / (global_step + 1), alpha)
        for t_p,p in zip(self.tea_params, self.params):
#            print('t_p: ', t_p)
            t_p.data.mul_(alpha)
            t_p.data.add_((1.0-alpha)*p.data)
 #           print('p: ', p)
  #          print('(after) t_p: ', t_p)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
       return 1.0
    else:
       current = np.clip(current, 0.0, rampup_length)
       phase = 1.0 - current / rampup_length
       return float(np.exp(-5.0*phase*phase))

def get_current_consistency_weight(weight, epoch, rampup):
    out = weight * sigmoid_rampup(epoch, rampup)
    print('Consistency_weight: ', out)
    return out

def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)

