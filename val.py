import torch

torch.cuda.set_device(0)
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
from os import listdir
import random
import nibabel as nib
import argparse
from Dataset import ValDataset,CropTestPatches
from AmygNet3D_indep import AmygNet3D
from crf_4d import refine_softmax_crf
import itertools
import medpy.metric.binary as mmb

def crf(prob,imgArray,n_iter,use_2d=False,sdims_in=4,schan_in=3,compat_in=3):
    labelcrf = refine_softmax_crf(prob, imgArray,use_2d = False,
                                            n_iter=n_iter, sdims_in=4, schan_in=3, compat_in=3)
    return labelcrf

def CalDice(SegRes, Ref, seg_labels, ref_labels):
    Dice_array = []
    ASSD_array = []
    print(len(np.unique(SegRes)))
    if len(np.unique(SegRes)) > 3:
       print('Subnuclei!')
       seg_labels = [1,2,3,4,5,6,7,8,9,10]
       ref_labels = [1,2,3,4,5,6,7,8,9,10]
       for res_c,ref_c in zip(seg_labels,ref_labels):
           if (res_c != 1) & (res_c != 6) & (ref_c != 1) & (res_c != 6):

              dc = mmb.dc(SegRes == res_c, Ref == ref_c)
              Dice_array.append(dc)
    else:
       print('Amygdala!')
       ref_labels = [1,2]
       seg_labels = [1,2]
       for res_c,ref_c in zip(seg_labels,ref_labels):
           dc = mmb.dc(SegRes == res_c, Ref == ref_c)
           Dice_array.append(dc)

    return Dice_array

def generate_indexes(patch_shape, expected_shape, pad_shape=[26,26,26]) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * ((expected_shape[i]-pad_shape[i]*2) // patch_shape[i+1]) + pad_shape[i]*2 for i in range(ndims-1)]
    idxs = [range(pad_shape[i], poss_shape[i] - pad_shape[i], patch_shape[i+1]) for i in range(ndims-1)]
    return itertools.product(*idxs)


def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    reconstructed_img = np.zeros((191,236,171))

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img


def val(val_loader,model,args):
    model.eval()
  #  for ema_params in model.parameters():
 #	 print('ema_params: ', ema_params.data[0])
  #         print('params: ', params.data[0])
  #         print(torch.eq(ema_params,params))



    mean_dice, mean_assd = [],[]
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['images'].float().cuda()
            label = sample['labels'].data.cpu().numpy()
            name = sample['names'][0]
            imgID = name.split(".")[0]

            img_patches = CropTestPatches(image.data.cpu().numpy(),args.patch_size,args.extraction_step)
            img_patches = Variable(img_patches).cuda()

            img_patches = img_patches.contiguous().view([-1,1] + args.patch_size).cuda()

            # B,C,H,W,D
            out = model(img_patches,args)
            out = out.permute(0,2,3,4,1).contiguous().cuda()
#            vol = nib.load('../ISMRM_Dataset/Training/subject208.nii')

            segmentation = np.array(np.zeros([191,236,171]), dtype="int16")

            out = torch.max(out,4)[1].cuda()
            segmentation = reconstruct_volume(out.data.cpu().numpy(),[191,236,171])

            Dice_array = CalDice(segmentation, label, args.res_labels, args.ref_labels)
            mean_dice.append(np.mean(Dice_array))

            if args.num_classes == 3:

               print('{} | Mean dice: {:.4f}, leftamyg dice: {:.4f}, rightamyg dice: {:.4f}'.format(name,np.mean(Dice_array), Dice_array[0], Dice_array[1]))

            else:

               for i in xrange(0,len(Dice_array)):
                   print('Dice score of class_{}: {:.4f}'.format(i+1, Dice_array[i]))
               print('{} | ******************* Mean dice: {:.4f}:'. format(name, np.mean(Dice_array)))


 #           mean_assd.append(np.mean(ASSD_array))

      #      print('{} | Mean dice: {:.4f}, leftamyg dice: {:.4f}, rightamyg dice: {:.4f}'.format(name,np.mean(Dice_array), Dice_array[0], Dice_array[1]))
  #          print('{} | Mean assd: {:.4f}, leftamyg assd: {:.4f}, rightamyg assd: {:.4f}'.format(name,np.mean(ASSD_array), ASSD_array[0], ASSD_array[1]))

        if np.mean(mean_dice) > args.best_mean:
           args.best_epoch = args.epoch
           args.best_mean = np.mean(mean_dice)
           print('Best Mean: {:.4f}'.format(args.best_mean))


def main(args):

    def create_model():

        model = AmygNet3D(args.num_classes, args.wrs_ratio, args.drop_rate, args.wrs_ratio_fc, args.drop_rate_fc, args.test_state)
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
        cudnn.benchmark = True

  #	 if ema:
 #          for param in model.parameters():
#               param.detach_()
        return model

    model = create_model()
    ema_model = create_model()


    # collect the number of parameters in the network
    print("------------------------------------------")

    if os.path.isfile(args.epoch):
       print("=> Loading checkpoint '{}'".format(args.epoch))
       checkpoint = torch.load(args.epoch)
#	ema_model.load_state_dict(checkpoint['ema_state_dict'])
       model.load_state_dict(checkpoint['state_dict'])
       print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
       raise Exception("=> No checkpoint found at '{}'".format(args.epoch))

    vf = ValDataset(args)
    val_loader = DataLoader(vf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=False)

    if args.ema_val:
  #     for ema_params, params in zip(ema_model.parameters(),model.parameters()):
#           print('ema_params: ', ema_params.data[0])

 #          print('params: ', params.data[0])
  #         print(torch.eq(ema_params,params))

#	print('***********************************')
       val(val_loader, ema_model, args)

    else:
       val(val_loader, model, args)

if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('true', '1'):
           return True
        elif v.lower() in ('false', '0'):
           return False
        else:
           raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()

    # Path related arguments
    parser.add_argument('--val_path', default='/study/utaut2/Yilin/ISMRM_Dataset/GAN_SelfEnsembling')
    parser.add_argument('--valimagefolder', default='trans_TBI')
    parser.add_argument('--vallabelfolder', default='labels')
    parser.add_argument('--train_path', default='/study/utaut2/Yilin/ISMRM_Dataset/Training')
    parser.add_argument('--ckpt', default='./checkpoints',
                        help='folder to output checkpoints')
    parser.add_argument('--model', default='AmygNet')
    # Data related arguments
    parser.add_argument('--drop_rate',default=0,type=float)
    parser.add_argument('--wrs_ratio',default=1,type=float)
    parser.add_argument('--drop_rate_fc',default=0,type=float)
    parser.add_argument('--wrs_ratio_fc',default=1,type=float)
    parser.add_argument('--patch_size', default=[105,105,105], nargs='+',type=int)
    parser.add_argument('--center_size', default=[73,73,73], nargs='+',type=int)
    parser.add_argument('--extraction_step', default=[53, 53, 53], nargs='+',type=int)
    parser.add_argument('--num_classes', default=11,type=int)
    parser.add_argument('--res_labels', default=[1,2],nargs='+',type=int)
    parser.add_argument('--ref_labels', default=[1,2],nargs='+',type=int)
    parser.add_argument('--num_workers',default=20,type=int)
    parser.add_argument('--shuffle',default=False,type=bool)
    parser.add_argument('--norm_type', default='self',help='options:group,self,none')

    # val related arguments
    parser.add_argument('--num_gpus',default=1,type=int)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--start_epoch',default=1,type=int)
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='epochs for validation')
    parser.add_argument('--steps_spoch', default=10, type=int,
                        help='number of epochs to do the validation')
    parser.add_argument('--end_epoch', default=20, type=int,
                        help='The last epoch for validation')
    parser.add_argument('--best_epoch', default=0, type=int,
                        help='The epoch that has the best validation result')
    parser.add_argument('--best_mean', default=0.0, type=float,
                        help='the best mean dice score')
    parser.add_argument('--ema_val',default=False,type=str2bool)
    parser.add_argument('--test_state',default=True,type=str2bool)
    parser.add_argument('--triple',default=False,type=str2bool)

    args = parser.parse_args()
    print("input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key,value))

    args.ckpt = os.path.join(args.ckpt, args.model)

    args.ckpt_list = listdir(args.ckpt)
    args.ckpt_list.sort(key=lambda f: int(filter(str.isdigit, f)))

    for epoch in args.ckpt_list[:]:
        args.epoch = args.ckpt + '/' + str(epoch)
        main(args)

    print('Best Epoch: {}, Best mean: {:.4f}'.format(args.best_epoch, args.best_mean))

