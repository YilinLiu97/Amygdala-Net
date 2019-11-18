import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import random
import nibabel as nib
import argparse
from Dataset import TestDataset,CropTestPatches
from AmygNet3D_multi import AmygNet3D
from crf_4d import refine_softmax_crf
import itertools

def crf(prob,imgArray,n_iter,use_2d=False,sdims_in=4,schan_in=3,compat_in=3):
    labelcrf = refine_softmax_crf(prob, imgArray,use_2d = False,
                                            n_iter=n_iter, sdims_in=4, schan_in=3, compat_in=3)
    return labelcrf

def save_vol(segmentation,imageID,loc, crf_n_iter,use_crf=True):
    if not os.path.exists(loc):
       os.mkdir(loc)
    if not use_crf:
       imageName = '{0}/{1}.nii.gz'.format(loc,imageID+'_SegRes' )
    else:
       vol = nib.load(os.path.join(args.test_path, imageID+'.nii'))
       segmentation = crf(np.array(segmentation),vol.get_data(),n_iter=crf_n_iter)
       imageName = '{0}/{1}.nii.gz'.format(loc,imageID+'_Crf_SegRes' )

    niiToSave = nib.Nifti1Image(segmentation.astype('uint8'),None)
    nib.save(niiToSave,imageName)

    print("... Image succesfully saved in ", imageName)


def generate_indexes(patch_shape, expected_shape, pad_shape=[26,26,26]) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * ((expected_shape[i]-pad_shape[i]*2) // patch_shape[i+1]) + pad_shape[i]*2 for i in range(ndims-1)]
    print('poss_shape: ', poss_shape)
    idxs = [range(pad_shape[i], poss_shape[i] - pad_shape[i], patch_shape[i+1]) for i in range(ndims-1)]
    print('idxs: ', idxs)
    return itertools.product(*idxs)


def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    reconstructed_img = np.zeros(tuple(expected_shape))

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        print('selection: ', selection)
        reconstructed_img[selection] = patches[count]

    return np.flipud(reconstructed_img)



def test(test_loader,model,args):
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['images'].float().cuda()
            name = sample['name'][0]
            imgID = name.split(".")[0]

            print(image.size())
            img_patches = CropTestPatches(image.data.cpu().numpy(),args.patch_size,args.extraction_step)
            print(img_patches.size())        
            img_patches = Variable(img_patches).cuda()

            img_patches = img_patches.contiguous().view([-1,1] + args.patch_size).cuda()
    
            print(img_patches.size())
            # B,C,H,W,D
            out = model(img_patches, args)
            out = out.permute(0,2,3,4,1).contiguous().cuda()
            print('out.shape: ', out.size())
            vol = nib.load('/study/utaut2/Yilin/ISMRM_Dataset/Training/subject208.nii')

            segmentation = np.array(np.zeros(list(vol.get_shape())), dtype="int16")

            if args.crf:
               softout = np.array([t.data.cpu().numpy() for t in out])
               print('softout.shape: ',softout.shape)
               segmentation_all = []
               for c in range(args.num_classes):
                   smArray = reconstruct_volume(softout[:,:,:,:,c], args.expected_recon_shape)
                   segmentation_all.append(smArray)
               print('segmentation_all.shape: ', np.array(segmentation_all).shape)
               save_vol(segmentation_all,imgID,loc=args.result_path,crf_n_iter=args.crf_n_iter,use_crf=args.crf)

            else:
               out = torch.max(out,4)[1].cuda()
               print(out.size())
               segmentation = reconstruct_volume(out.data.cpu().numpy(), args.expected_recon_shape)

        #    print('label1.sum(): ',np.sum(out.data.cpu().numpy()==1))
        #    print('label2.sum(): ',np.sum(out.data.cpu().numpy()==2))

               save_vol(segmentation,imgID,loc=args.result_path,crf_n_iter=args.crf_n_iter,use_crf=args.crf)
         
        print('------------ Evaluation Done! --------------')


def test_AdaBN(test_loader,model,args):
   # model.train() # to not use running mean/variance.

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['images'].float().cuda()
            name = sample['name'][0]
            imgID = name.split(".")[0]
            pop_mean, pop_std = sample['pop_mean'].float().cuda(), sample['pop_std'].float().cuda()

            for m in model.modules():
                if isinstance(m, nn.BatchNorm3d):
                   m.weight = nn.Parameter(pop_mean)
                   m.bias = nn.Parameter(pop_std)

            print(image.size())
            img_patches = CropTestPatches(image.data.cpu().numpy(),args.patch_size,args.extraction_step)
            print(img_patches.size())
            img_patches = Variable(img_patches).cuda()

            img_patches = img_patches.contiguous().view([-1,1] + args.patch_size).cuda()

            print(img_patches.size())
            # B,C,H,W,D
            out = model(img_patches,args.patch_size)
            out = out.permute(0,2,3,4,1).contiguous().cuda()
            print('out.shape: ', out.size())
            vol = nib.load('../ISMRM_Dataset/Training/subject205.nii')

            segmentation = np.array(np.zeros(list(vol.get_shape())), dtype="int16")

            if args.crf:
               softout = np.array([t.data.cpu().numpy() for t in out])
               print('softout.shape: ',softout.shape)
               segmentation_all = []
               for c in range(args.num_classes):
                   smArray = reconstruct_volume(softout[:,:,:,:,c], args.expected_recon_shape)
                   segmentation_all.append(smArray)
               print('segmentation_all.shape: ', np.array(segmentation_all).shape)
               save_vol(segmentation_all,imgID,loc=args.result_path,crf_n_iter=args.crf_n_iter,use_crf=args.crf)

            else:
               out = torch.max(out,4)[1].cuda()
               print(out.size())
               segmentation = reconstruct_volume(out.data.cpu().numpy(),args.expected_recon_shape)

               save_vol(segmentation,imgID,loc=args.result_path,crf_n_iter=args.crf_n_iter,use_crf=args.crf)

        print('------------ Evaluation Done! --------------')



def main(args):

    model = AmygNet3D(args.num_classes, args.wrs_ratio, args.drop_rate, args.wrs_ratio_fc, args.drop_rate_fc, args.test_state)
    model = nn.DataParallel(model,device_ids=list(range(args.num_gpus))).cuda()

    cudnn.benchmark = True

    # collect the number of parameters in the network
    print("------------------------------------------")
    num_para = 0
    for name,param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print(model)
    print("Number of trainable parameters %d in Model %s" % (num_para, 'AmygNet'))
    print("------------------------------------------")

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.ema_test:
               state_dict = checkpoint['ema_state_dict']
            else:
               state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            raise Exception("=> No checkpoint found at '{}'".format(args.resume))

    tf = TestDataset(args)
    test_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=False)

    if not args.use_AdaBN:
       test(test_loader, model, args)
    else:
       test_AdaBN(test_loader, model, args)

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
    parser.add_argument('--test_path', default='/study/utaut2/Yilin/ISMRM_Dataset/Testing/translated_TBI_patients')
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
    parser.add_argument('--extraction_step', default=[53, 53, 53], nargs='+',type=int)
    parser.add_argument('--num_classes', default=3,type=int)
    parser.add_argument('--num_workers',default=20,type=int)
    parser.add_argument('--shuffle',default=False,type=bool)
    parser.add_argument('--norm_type', default='self',help='options: group, self, none')

    # Test related arguments
    parser.add_argument('--num_gpus',default=1,type=int)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--test_epoch',default=360,type=int)
    parser.add_argument('--save_path',default='AmygNet',help='folder to save prediction results')
    parser.add_argument('--num_round',default=None, type=int, help='restore the models from which run')
    parser.add_argument('--crf',default=False,type=bool)
    parser.add_argument('--crf_n_iter',default=5,type=int)
    parser.add_argument('--use_AdaBN',default=False,type=bool)
    parser.add_argument('--ema_test',default=False,type=bool)
    parser.add_argument('--test_state',default=True,type=str2bool)
    parser.add_argument('--triple',default=False,type=str2bool)
    parser.add_argument('--args.expected_recon_shape', default=[191,236,171], nargs='+',type=int)

    args = parser.parse_args()
    print("input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key,value))

    if not args.num_round:
        args.ckpt = os.path.join(args.ckpt, args.model)
    else:
	args.ckpt = os.path.join(args.ckpt, args.model, str(args.num_round))
    
    args.result_path = 'SegResults' + '/' +  args.save_path

    if not os.path.isdir(args.result_path):
       os.makedirs(args.result_path)

    args.resume = args.ckpt + '/' + str(args.test_epoch) + '_checkpoint.pth.tar'
    main(args)

