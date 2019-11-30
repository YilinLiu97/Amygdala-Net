from torch.utils.data import Dataset
import random
import numpy as np
import nibabel as nib
import torch
import os
from os import listdir,path
from os.path import join
from collections import defaultdict
from sklearn.feature_extraction.image import extract_patches
import imgaug as ia
from imgaug import augmenters as iaa

class MakeTrainData(object):
    def __init__(self,patch_size,num_patches,num_classes,norm_type,is_sup):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.is_sup = is_sup
        self.norm_type = norm_type

    def __call__(self,folder1_path,folder2_path):

        if self.is_sup:
           TrainData_path = folder1_path
           print(TrainData_path)
           TrainData_dir = listdir(TrainData_path)
           TrainData_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

           if self.norm_type == 'group':
              _,self.mean,self.std = normalization(TrainData_path)

           else:
              self.mean, self.std = 0,1 #std = 1 to avoid 'divide-by-0' error

           LabelData_path = folder2_path
           LabelData_dir = listdir(LabelData_path)
           LabelData_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

           img_patches, gt_patches = [],[]
           for image,label in zip(TrainData_dir, LabelData_dir):
               image = nib.load(join(TrainData_path,image)).get_data()
               label = nib.load(join(LabelData_path,label)).get_data()

               if self.norm_type == 'self':
                  self.mean, self.std = np.mean(image), np.std(image)

               image = (image - np.mean(image))/np.std(image)

               sample = {'images': image, 'labels':label, 'targets': None}
               transform = CropPatches(self.patch_size,self.num_patches,self.num_classes)
               imgs,gts = transform(sample) # patches cropped from a single subject

               imgs_aug, gts_aug = Simple_Aug(imgs, gts)

               img_patches.append(imgs)
               gt_patches.append(gts)

           img_patches = np.asarray(img_patches).reshape(-1,59,59,59)
           gt_patches = np.asarray(gt_patches).reshape(-1,59,59,59)

           return np.asarray(img_patches),np.asarray(gt_patches)

        else:
           TargetData_orig_path = folder1_path
           print(TargetData_orig_path)
           TargetData_trans_path = folder2_path

           TargetData_orig_dir = listdir(TargetData_orig_path)
           TargetData_orig_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

           TargetData_trans_dir = listdir(TargetData_trans_path)
           TargetData_trans_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

           target_orig_patches,target_trans_patches = [],[]
           for origname,transname in zip(TargetData_orig_dir,TargetData_trans_dir):
               target_orig = nib.load(join(TargetData_orig_path,origname)).get_data()
               target_trans = nib.load(join(TargetData_trans_path,transname)).get_data()

               if self.norm_type == 'self':
                  self.orig_mean, self.orig_std = np.mean(target_orig), np.std(target_orig)
                  self.trans_mean, self.trans_std = np.mean(target_trans), np.std(target_trans)

               target_orig = (target_orig-self.orig_mean)/self.orig_std
               target_trans = (target_trans-self.trans_mean)/self.trans_std

               target_orig = CropTargetPatches(target_orig,self.patch_size,extraction_step=[27,27,27])
               target_trans = CropTargetPatches(target_trans,self.patch_size,extraction_step=[27,27,27])

               target_orig_patches.append(target_orig)
               target_trans_patches.append(target_trans)


           target_orig_patches = np.asarray(target_orig_patches).reshape(-1,59,59,59)
           target_trans_patches = np.asarray(target_trans_patches).reshape(-1,59,59,59)
           print(np.array(target_orig_patches).shape)
           print(np.array(target_trans_patches).shape)
           return np.asarray(target_orig_patches), np.asarray(target_trans_patches)

class TrainDataset(Dataset):
    def __init__(self,args):
        self.data_path = args.data_path
        self.source_path = join(args.data_path,args.sourcefolder)
        self.label_path = join(args.data_path,args.labelfolder)

        self.patch_size = args.patch_size
        self.num_patches = args.num_patches
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.norm_type = args.norm_type

        prepare_traindata = MakeTrainData(self.patch_size,self.num_patches,self.num_classes,self.norm_type,is_sup=True)
        self.img_patches, self.gt_patches = prepare_traindata(self.source_path,self.label_path)
        print('%%%%%%%%%%%%%%%% ', self.img_patches.shape)
        print('%%%%%%%%%%%%%%%% ', self.gt_patches.shape)
        self.length = len(self.img_patches)


    def __len__(self):
        return self.length

    def __getitem__(self,idx):

        return {'images':self.img_patches[idx], 'labels':self.gt_patches[idx]}


class Target_TrainDataset(Dataset):
    def __init__(self,args):
        self.data_path = args.data_path
        self.target_trans_path = join(args.data_path,args.target_trans_folder)
        self.target_orig_path = join(args.data_path,args.target_orig_folder)

        self.patch_size = args.patch_size
        self.num_patches = args.num_patches
        self.num_classes = args.num_classes
        self.norm_type = args.norm_type

        prepare_targetdata = MakeTrainData(self.patch_size,self.num_patches,self.num_classes,self.norm_type,is_sup=False)
        self.target_orig_patches, self.target_trans_patches = prepare_targetdata(self.target_orig_path,self.target_trans_path)
        self.length = len(self.target_orig_patches)

        c = list(zip(self.target_orig_patches, self.target_trans_patches))
        random.shuffle(c)
        self.target_orig_patches, self.target_trans_patches = zip(*c)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):

        return self.target_orig_patches[idx], self.target_trans_patches[idx]

class ValDataset(Dataset):
    def __init__(self,args):
       self.Val_path = args.val_path
       self.Val_data_path = join(self.Val_path,args.valimagefolder)
       self.Val_labels_path = join(self.Val_path,args.vallabelfolder)

       self.Val_data_dir = listdir(self.Val_data_path)
       self.Val_labels_dir = listdir(self.Val_labels_path)

       self.Val_data_dir.sort(key=lambda f: int(filter(str.isdigit, f)))
       self.Val_labels_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

       self.length = len(self.Val_data_dir)
       self.norm_type = args.norm_type

       if self.norm_type == 'group':
          _,self.mean,self.std = normalization(args.train_path)
       else:
          self.mean, self.std = 0,1


    def __len__(self):
       return self.length

    def __getitem__(self,idx):
        image = nib.load(join(self.Val_data_path,self.Val_data_dir[idx])).get_data()
        label = nib.load(join(self.Val_labels_path,self.Val_labels_dir[idx])).get_data()
        name = self.Val_data_dir[idx]
        label = np.asarray(label)

        if self.norm_type == 'self':
           self.mean, self.std = np.mean(image), np.std(image)

        # volume-wise intensity normalization
        image = (image - self.mean) / self.std
        sample = {'images': image, 'labels': label.astype(int), 'names': name}
        return sample

class TestDataset(Dataset):
    def __init__(self,args):
        self.TestData_path = args.test_path
        self.TestData_dir = listdir(self.TestData_path)
        self.length = len(listdir(self.TestData_path))
        self.norm_type = args.norm_type
        self.TestData_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

        if self.norm_type == 'group':
           _,self.mean,self.std = normalization(args.train_path)
        else:
           self.mean, self.std = 0,1


    def __len__(self):
        return self.length

    def __getitem__(self,idx):

        image = nib.load(join(self.TestData_path,self.TestData_dir[idx])).get_data()

        if self.norm_type == 'self':
           self.mean,self.std = np.mean(image), np.std(image)

        # volume-wise intensity normalization
        image = (image - self.mean) / self.std
        name = self.TestData_dir[idx]

        sample = {'images':image,'name':name, 'pop_mean':self.mean, 'pop_std':self.std}
        return sample


class CropPatches(object):
    def __init__(self,patch_size,num_patches,num_classes):

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_patches = num_patches

    def __call__(self,sample):
        image,label = sample['images'], sample['labels']
        if sample['targets'] is not None:
           target = sample['targets']
           targets = []

        h,w,d = self.patch_size

        # generate the training batch with equal probability for each class
        fb = np.random.choice(2)
        if fb:
           index = np.argwhere(label > 0)
        else:
           index = np.argwhere(label == 0)

        # randomly choose N center position
        choose = random.sample(range(0,len(index)),self.num_patches)
        centers = index[choose].astype(int)

        images, gts = [],[]
        # check whether the left and right index overflow
        for center in centers:
            left = []
            for i in range(3):
                margin_left = int(self.patch_size[i]/2)
                margin_right = self.patch_size[i] - margin_left

                left_index = center[i] - margin_left
                right_index = center[i] + margin_right

                if left_index < 0:
                   left_index = 0
                if right_index > label.shape[i]:
                   left_index = left_index - (right_index - label.shape[i])
                left.append(left_index)

            img = np.zeros([h,w,d])
            gt = np.zeros([h,w,d])

            img = image[left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]
            gt = np.array(label[left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d])

            images.append(img)
            gts.append(gt)

            if sample['targets'] is not None:
               tg = target[left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]
               targets.append(tg)

        if sample['targets'] is not None:
           return np.asarray(images), np.asarray(targets)
        else:
           return np.asarray(images), np.asarray(gts)

'''
Augmentation: Only do scaling and flipping.
'''
def Simple_Aug(img_patches, gt_patches):
    seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8,1.2),
        rotate=4.6),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
])
    if gt_patches is None:
       return seq.augment_images(img_patches)
    else:
       img_patches_aug = seq.augment_images(img_patches)
       gt_patches_aug = seq.augment_images(gt_patches)

    return img_patches_aug, gt_patches_aug


def Augmentation(img_patches, gt_patches):
    seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8,1.2),
        translate_percent=0.03,
        rotate=4.6),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images  #alpha: the strength of the displacement. sigma: the smoothness of the displacement.
    iaa.ElasticTransformation(alpha=(28.0, 30.0), sigma=3.5)
])                                                                  #suggestions - alpha:sigma = 10 : 1
    if gt_patches is None:
       return seq.augment_images(img_patches)
    else:
       img_patches_aug = seq.augment_images(img_patches)
       gt_patches_aug = seq.augment_images(gt_patches)

    return img_patches_aug, gt_patches_aug

def Crop(img, out_size):
    _,_,a,b,c = img.size()
    patch_size = [a,b,c]

    start_index = []
    end_index = []
    for i in range(3):

        start = int((patch_size[i] - out_size[i])/2)
        start_index.append(start)
        end_index.append(start + out_size[i])

    img = img[:,:,start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]
    return img

def CropTargetPatches(img,patch_size,extraction_step):
    crop_diff = 16
    out_size = patch_size[0] - 2*crop_diff

    img_patches = extract_patches(img,patch_size,tuple(extraction_step))

    return img_patches

def CropTestPatches(img,patch_size,extraction_step):
    img = np.squeeze(img,axis=0)
    crop_diff = 16
    out_size = patch_size[0] - 2*crop_diff

    img_patches = extract_patches(img,patch_size,tuple(extraction_step))

    return torch.from_numpy(img_patches)

def normalization(Data_path):
    Data_dir = listdir(Data_path)
    Data_dir.sort(key=lambda f: int(filter(str.isdigit, f)))

    Vols = np.empty((len(Data_dir),191,236,171))
    for idx in range(0,len(Data_dir)):
        Vols[(idx),:,:,:] = nib.load(join(Data_path,listdir(Data_path)[idx])).get_data()

    Vols_mean = Vols.mean()
    Vols_std = Vols.std()
    Vols = (Vols - Vols_mean) / Vols_std
    return Vols, Vols_mean, Vols_std

