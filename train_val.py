import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from Dataset import *
import os
from os import listdir
from os.path import isfile,join
import nibabel as nib
import argparse
from utils import AverageMeter,get_current_consistency_weight
from losses import softmax_mse_loss, softmax_kl_loss
import math
from AmygNet3D_multi import AmygNet3D
from imgaug import augmenters as iaa
import medpy.metric.binary as mmb
import itertools


def Online_Augmentation(inputs, unsup_preds):
    seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8,1.2),
        translate_percent=0.03,
        rotate=4.6),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.ElasticTransformation(alpha=(28.0, 30.0), sigma=3.5) #alpha: the strength of the displacement. sigma: the smoothness of the displacement.
])                                                                  #suggestions - alpha:sigma = 10 : 1
    inputs_size = inputs.size()[0]
    inputs_numpy = inputs.squeeze(1).cpu().detach().numpy()
    unsup_preds_numpy = unsup_preds.cpu().detach().numpy()

    img_patches_aug = seq.augment_images(inputs_numpy)

    gt_patches_aug = []
    for c in range(args.num_classes):
        gt_patches_aug.append(seq.augment_images(unsup_preds_numpy[:,c,:,:,:]))

    gt_patches_aug = np.asarray(gt_patches_aug)

    return torch.from_numpy(img_patches_aug).unsqueeze(1).cuda(), torch.from_numpy(gt_patches_aug).permute(0,2,3,4,1).contiguous().view(-1,args.num_classes).float().cuda()

def CalDice(SegRes, Ref, seg_labels, ref_labels):
    Dice_array = []

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

def validate(model, ema_model, val_loader, args):

    model.eval()

    if not args.sup_only:
       ema_model.eval()

#    load_bn_params(model, ema_model)
 #   compare_models(ema_model,model)

    mean_dice, ema_mean_dice = [],[]
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['images'].float().cuda()
            label = sample['labels'].data.cpu().numpy()
            name = sample['names'][0]
            imgID = name.split(".")[0]

            img_patches = CropTestPatches(image.data.cpu().numpy(),[105,105,105],[53,53,53])
            img_patches = Variable(img_patches).cuda()

            img_patches = img_patches.contiguous().view([-1,1] + [105,105,105]).cuda()

            # B,C,H,W,D
            out = model(img_patches,[105,105,105])
            out = out.permute(0,2,3,4,1).contiguous().cuda()

            if not args.sup_only:
               ema_out = ema_model(img_patches,[105,105,105])
               ema_out = ema_out.permute(0,2,3,4,1).contiguous().cuda()

            vol = nib.load('../ISMRM_Dataset/Training/subject208.nii')
            segmentation = np.array(np.zeros(list(vol.get_shape())), dtype="int16")

            out = torch.max(out,4)[1].cuda()
            segmentation = reconstruct_volume(out.data.cpu().numpy(),[191,236,171])

            Dice_array = CalDice(segmentation, label, args.res_labels, args.ref_labels)
       	    mean_dice.append(np.mean(Dice_array))

            if not args.sup_only:
               ema_out = torch.max(ema_out,4)[1].cuda()
               ema_segmentation = reconstruct_volume(ema_out.data.cpu().numpy(),[191,236,171])

               ema_Dice_array = CalDice(ema_segmentation, label, args.res_labels, args.ref_labels)
               ema_mean_dice.append(np.mean(ema_Dice_array))

            print('{} | Mean dice: {:.4f}'.format(name,np.mean(Dice_array)))

            if not args.sup_only:
               print('{} | ema Mean dice: {:.4f}'.format(name,np.mean(ema_Dice_array)))


def train(train_loader, target_train_loader, model, ema_model, Sup_criterion, consistency_criterion, consistency_weight, rampup_alpha, optimizer, epoch, args):
    losses = AverageMeter()
    Sup_losses = AverageMeter()
    UnSup_losses = AverageMeter()

    global_step = 0

    #Train mode
    model.train()
    if not args.sup_only:
       ema_model.train()
       target_loader = iter(target_train_loader)

    for iteration, sample in enumerate(train_loader):

        image = sample['images']
        label = sample['labels']

        image = Variable(image.unsqueeze(1)).float().cuda()
        label = Variable(label).long()

        # Dimension of the output: B,C,W,H,D. Transform the prediction
        out = model(image,args)
        out = out.permute(0,2,3,4,1)
        out_reshape = out.contiguous().view(-1,args.num_classes)

        # extract the center part of the labels
        start_index = []
        end_index = []
        for i in range(3):
            start = int((args.patch_size[i] - args.out_size[i])/2)
            start_index.append(start)
            end_index.append(start + args.out_size[i])

        label = label[:,start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]

        label = Variable(label).cuda()
        label = label.contiguous().view(-1).cuda()

        Sup_loss = Sup_criterion(out_reshape,label)  # Supervised component

        # ------------- Unsupervised component (Calculate consistency for unlabeled samples only)------------- #
        if not args.sup_only:
           target_orig_inputs, target_trans_inputs = target_loader.next()
           target_orig_inputs = target_orig_inputs.contiguous().view([-1,1] + args.patch_size).float().cuda()
           target_trans_inputs = target_trans_inputs.contiguous().view([-1,1] + args.patch_size).float().cuda()

           # Teacher forward
           with torch.no_grad():
                teacher_target_out = ema_model(target_trans_inputs,args.patch_size)

           Augmenter = Tensor_Augmenter()
           target_inputs_aug, teacher_target_aug_out = Augmenter(target_orig_inputs,teacher_target_out)


           # Student forward
           student_target_out = model(target_inputs_aug,args.patch_size)
           student_target_out = student_target_out.permute(0,2,3,4,1).contiguous().view(-1,args.num_classes).cuda()

           consistency_loss = consistency_weight * consistency_criterion(student_target_out,teacher_target_aug_out)

           # Total loss
           loss = Sup_loss  + consistency_loss

           Sup_losses.update(Sup_loss.data[0],target_orig_inputs.size(0))
           UnSup_losses.update(consistency_loss.data[0],target_orig_inputs.size(0))

        else:
           loss = Sup_loss

        losses.update(loss.data[0],image.size(0))


        global_step += 1

        # compute gradient and do SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not args.sup_only:
           # update teacher model
           if epoch <= args.consistency_rampup_epoch:
              update_ema_variables(model, ema_model, rampup_alpha, global_step)
           else:
              update_ema_variables(model, ema_model, args.alpha_late, global_step)

           print('   * i {} |  lr: {:.6f} | Sup_Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr, losses=Sup_losses))
           print('   * i {} |  lr: {:.6f} | UnSup_Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr, losses=UnSup_losses))

        # adjust learning rate
        cur_iter = iteration + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, args)

        print('   * i {} |  lr: {:.6f} | Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr, losses=losses))

    print('   * EPOCH {epoch} | Training Loss: {losses.avg:.3f}'.format(epoch=epoch, losses=losses))
    
    if epoch >= args.particular_epoch:
            if epoch % args.val_every_n_epoch == 0:
               print('*********** Validating **************')
               vf = ValDataset(args)
               val_loader = DataLoader(vf, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
               if not args.sup_only:
                  validate(model, ema_model, val_loader, args)
               else:
                  validate(model, None, val_loader, args)

def save_checkpoint(state, epoch, args):
    filename = args.ckpt + '/' + str(epoch) + '_checkpoint.pth.tar'
    print(filename)
    torch.save(state, filename)

def adjust_learning_rate(optimizer, cur_iter, args):
    print('cur_iter: ', cur_iter)
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    print('alpha: ', alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    

def load_bn_params(model, ema_model):
    for childA, childB in zip(ema_model.children(), model.children()):
        if isinstance(childB, nn.BatchNorm3d):
           childA.running_mean = childB.running_mean
           childA.running_var = childB.running_var


'''
def load_bn_params(model,ema_model):
    ema_state = ema_model.state_dict()
    type(ema_state)
    ema_items = ema_model.state_dict().items()
    model_items = model.state_dict().items()

    bn_params = {k: v for k,v in model_items if not torch.equal(v,ema_state[k])}
    ema_state.update(bn_params)
    ema_model.load_state_dict(ema_state)
'''
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
#                print('correcting....')
 #               key_item_1[1] = key_item_2[1]
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def main(args):

    def create_model(ema=False):

        model = AmygNet3D(args.num_classes, args.wrs_ratio, args.drop_rate, args.wrs_ratio_fc, args.drop_rate_fc)
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
        cudnn.benchmark = True

        if ema:
           for param in model.parameters():
               param.detach_()

        return model

    model = create_model()

    # UnSupervised mode
    if not args.sup_only:
       ema_model = create_model(ema=True)
       ema_model.load_state_dict(model.state_dict()) # deep copy
       
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

    # set the optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), args.lr)
    Sup_criterion = nn.CrossEntropyLoss()
    if not args.sup_only:
       consistency_criterion = softmax_mse_loss
       print('...MSE Loss for Consistency Calculation')

    # Resume training
    if args.resume:
        if os.path.isfile(args.resume_epoch):
           print("=> Loading checkpoint '{}'".format(args.resume_epoch))
           checkpoint = torch.load(args.resume_epoch)
           start_epoch = checkpoint['epoch']
           model.load_state_dict(checkpoint['state_dict'])
           if not args.sup_only:
              ema_model.load_state_dict(checkpoint['ema_state_dict'])
           optimizer.load_state_dict(checkpoint['opt_dict'])
           print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
           print("=> No checkpoint found at '{}'".format(args.resume_epoch))

    print('Start training ...')

    # Sample all target brains only once. No Shuffle!!
    if not args.sup_only:
       target_data = Target_TrainDataset(args)
       target_loader = DataLoader(target_data,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)

    for epoch in range(args.start_epoch + 1, args.num_epochs + 1):

        # Re-sample from the whole brain after an epoch
        source_data = TrainDataset(args)
        train_loader = DataLoader(source_data,batch_size=args.batch_size,shuffle=args.shuffle,num_workers=args.num_workers,pin_memory=True)

        if not args.sup_only:
           consistency_weight = get_current_consistency_weight(args.cons_weight, epoch, args.consistency_rampup_epoch)
           rampup_alpha = args.alpha #get_current_consistency_weight(args.alpha, epoch, args.alpha_rampup_epoch)
           train(train_loader, target_loader, model, ema_model, Sup_criterion, consistency_criterion, consistency_weight, rampup_alpha,  optimizer, epoch, args)
        else:
           train(train_loader, None, model, None, Sup_criterion, None, None,None, optimizer, epoch, args)

        # save models
        if epoch >= args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                if not args.sup_only:
                   save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'ema_state_dict': ema_model.state_dict(), 'opt_dict': optimizer.state_dict()}, epoch, args)
                else:
                   save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, epoch, args)

    print("Training Done")

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
    parser.add_argument('--data_path', default='/study/utaut2/Yilin/ISMRM_Dataset/GAN_SelfEnsembling')
    parser.add_argument('--sourcefolder', default= 'Aug_training')
    parser.add_argument('--labelfolder', default= 'Aug_labels')
    parser.add_argument('--target_trans_folder', default= 'trans_TBI')
    parser.add_argument('--target_orig_folder', default= 'orig_TBI')
    parser.add_argument('--ckpt', default='./checkpoints')


    # Data related arguments
    parser.add_argument('--drop_rate_fc', default=0, type=float)
    parser.add_argument('--wrs_ratio_fc', default=1, type=float)
    parser.add_argument('--drop_rate', default=0, type=float)
    parser.add_argument('--wrs_ratio', default=1, type=float)
    parser.add_argument('--patch_size', default=[59,59,59],nargs='+', type=int)
    parser.add_argument('--center_size', default=[27,27,27],nargs='+', type=int)
    parser.add_argument('--num_patches', default=7, type=int)
    parser.add_argument('--out_size',default=[7,7,7],nargs='+', type=int)
    parser.add_argument('--num_classes',default=3, type=int)
    parser.add_argument('--num_workers',default=14,type=int)
    parser.add_argument('--experiment_name', default='GAN_SefEnsembling')
    parser.add_argument('--norm_type', default='self',help='options: group, self, none')

    # optimization related arguments
    parser.add_argument('--val_every_n_epoch', default=1, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=11, type=int)
    parser.add_argument('--num_epochs', default=4000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_pow', default=0.9, type=float)
    parser.add_argument('--save_epochs_steps', default=5, type=int)
    parser.add_argument('--particular_epoch', default=2, type=int)
    parser.add_argument('--shuffle', default=True,type=str2bool)
    parser.add_argument('--alpha', default=0.998, type=float)
    parser.add_argument('--cons_weight', default=1, type=float)
    parser.add_argument('--consistency_rampup_epoch', default=150, type=int)
    parser.add_argument('--sup_only', default=True,type=str2bool)
    parser.add_argument('--triple', default=True,type=str2bool)
    parser.add_argument('--mse_loss', default=True,type=str2bool)
   

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.ckpt = os.path.join(args.ckpt, args.experiment_name)
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume_epoch = args.ckpt + '/' + str(args.start_epoch) + '_checkpoint.pth.tar'
        args.resume = True
    else:
	args.resume = False

    args.running_lr = args.lr
    args.epoch_iters = math.ceil(28*args.num_patches/args.batch_size)
    args.max_iters = args.epoch_iters * args.num_epochs
  
    main(args)
