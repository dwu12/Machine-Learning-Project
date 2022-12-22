import cv2
import os
import math
import numbers
import random
import logging
import numpy as np
import imgaug
import imgaug.augmenters as iaa
import nibabel as nib

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
#from   torchvision import transforms
from scipy.ndimage import zoom

from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
CONFIG.data = EasyDict({})
CONFIG.model = EasyDict({})

CONFIG.data.real_world_aug = False
CONFIG.data.random_interp = True

CONFIG.model.crop_size=128
CONFIG.model.depth_size=32

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def get_unique_idx(fg):
    unique_idx=[]
    allitems=[]
    fgnew=[]
    for i in range(len(fg)):
        fi=fg[i]
        img_path, zmap_path, label_img_path, slice_num = fi
        if label_img_path not in allitems:
            unique_idx.append(i)
            allitems.append(label_img_path)
            fgnew.append(fi)
    return fgnew


class ImageFileTrain():
    def __init__(self,data_file):
        super(ImageFileTrain, self).__init__()

        self.alpha=get_unique_idx(np.load(data_file,allow_pickle=True).item()['train'])

    def __len__(self):
        return len(self.alpha)

class ImageFile():
    def __init__(self,data_file):
        super(ImageFile, self).__init__()

        self.alpha=get_unique_idx(np.load(data_file,allow_pickle=True).item()['val'])

    def __len__(self):
        return len(self.alpha)

class ImageFiletest():
    def __init__(self,data_file):
        super(ImageFiletest, self).__init__()

        print (data_file)
        self.alpha=get_unique_idx(np.load(data_file,allow_pickle=True).item()['test'])

    def __len__(self):
        return len(self.alpha)


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test", real_world_aug = False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase
        

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, = sample['fg'], sample['alpha']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        alpha = np.expand_dims(alpha, axis=0)

        sample['fg'], sample['alpha']= torch.from_numpy(image), torch.from_numpy(alpha),
        #sample['fg'] = sample['fg'].sub_(self.mean).div_(self.std)


        return sample


from skimage.exposure import equalize_adapthist
from scipy.stats import stats

def contrast_strech(img):
    if len(np.unique(img))<2:
        imgnew=np.zeros_like(img)
        return imgnew

    else:
        imgori = img.copy()

        img = img.astype(np.float32)
        # print ("unique check",np.unique(img))
        #print (len(np.unique(img)),np.unique(img))
        imgs = img.flatten()

        z = np.abs(stats.zscore(imgs))
        threshold = 2.5

        imgs = imgs[np.where(z <= threshold)]

        #print ("noew",imgs)
        norm_v=(np.max(imgs) - np.min(imgs))
        if norm_v>0:
            imgnew = (img - np.min(imgs)) / norm_v
            #print (np.min(imgnew),np.max(imgnew))
            imgnew[imgnew <=0] = 0
            imgnew[imgnew >= 1] = 1
            #imgnew=imgnew * 255
        else:
            imgnew=imgori
        #imgnew=np.asarray(imgnew,dtype=np.uint8)
        return imgnew


def regular_norm(imgs):


    norm_v=(np.max(imgs) - np.min(imgs))

    imgnew = (imgs - np.min(imgs)) / norm_v

    return imgnew


import scipy


def resize_data_volume_by_scale(data, scale_list):
   """
   Resize the data based on the provided scale
   """
   #scale_list = [scale,scale,scale]
   return scipy.ndimage.zoom(data, scale_list, order=0)



class DataGenerator(Dataset):
    def __init__(self, data, CONFIG, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.model.crop_size
        self.fg = data.alpha
        self.depth_size=CONFIG.model.depth_size



    def __getitem__(self, idx):

        #print (self.fg[idx])

        img_path, zmap_path, label_img_path, slice_num = self.fg[idx]
        #print (img_path)
        #img_path=img_path.replace("/neuro/labs/grantlab/research/SmallDiffuseLesionSegmentation/HIE_lesions/downloaded/","/Users/baorina/Downloads/visHIE/")
        #zmap_path=zmap_path.replace("/neuro/labs/grantlab/research/SmallDiffuseLesionSegmentation/HIE_lesions/downloaded/","/Users/baorina/Downloads/visHIE/")
        #label_img_path=label_img_path.replace("/neuro/labs/grantlab/research/SmallDiffuseLesionSegmentation/HIE_lesions/downloaded/","/Users/baorina/Downloads/visHIE/")
        #print (img_path)
        slice_num = int(slice_num)

        img_adc_all = nib.load(img_path).get_fdata()
        img_z_all = nib.load(zmap_path).get_fdata()

        bou_xy = np.asarray(np.where(img_adc_all > 0))
        # max_xy=np.asarray(np.where(test_image[:,:,i]>0))

        min_yx = np.min(bou_xy, 1)
        max_yx = np.max(bou_xy, 1)

        img_adc=img_adc_all[min_yx[0]:max_yx[0],min_yx[1]:max_yx[1],:]
        img_z=img_z_all[min_yx[0]:max_yx[0],min_yx[1]:max_yx[1],:]

        img_adc=regular_norm(img_adc)
        #img_z=regular_norm(img_z)
        label_image = nib.load(label_img_path).get_fdata()[min_yx[0]:max_yx[0],min_yx[1]:max_yx[1], :]
        gt_ori=label_image

        label_image = (label_image==1)

        data_in = np.concatenate([np.expand_dims(img_adc, 0), np.expand_dims(img_z, 0),  np.expand_dims(label_image,0)], 0)

        fa = self.depth_size / label_image.shape[-1]

        data_in_new=[]
        #print ("shape in shape",data_in.shape)
        scale_list=[data_in.shape[1]/self.crop_size,data_in.shape[2]/self.crop_size,data_in.shape[3]/self.depth_size]

        for i in range(data_in.shape[0]):
            #data_in_i=zoom(data_in[i,:,:,:], (self.crop_size/data_in.shape[1], self.crop_size/data_in.shape[2], fa))
            data_in_i=resize_data_volume_by_scale(data_in[i,:,:], [self.crop_size/data_in.shape[1],self.crop_size/data_in.shape[2],self.depth_size/data_in.shape[3]])
            #print ("data i ",data_in_i.shape)
            #data_in_i=(data_in[i,:,:,:])
            data_in_new.append(data_in_i)

        #print ("zoom back")
        data_in_new = np.asarray(data_in_new)
        #print ("data in new shape",data_in_new.shape)
        # data_back=[]
        # for i in range(data_in_new.shape[0]):
        #     data_in_i=zoom(data_in_new[i,:,:,:], (data_in.shape[1]/self.crop_size, data_in.shape[2]/self.crop_size, 1/fa))
        #     print ("data i ",data_in_i.shape)
        #     data_back.append(data_in_i)
        #
        # data_back=np.asarray(data_back)
        # alpha_back=1*(data_back[-1,:,:,:]>0.5)

        #print ("data back shape",data_back.shape)
        #print ("sumcheck",np.sum(np.abs(alpha_back-data_in[-1,:,:,:])))

        image=data_in_new[:-1,:,:,:]
        alpha=1*(data_in_new[-1,:,:,:]>0.1)

        alpha=np.asarray(alpha,dtype=np.float32)

        image_name = img_path

        #print ("shape check", image.shape,alpha.shape)
        image=torch.from_numpy(image)
        alpha=torch.from_numpy(alpha)
        image=torch.transpose(image, 1, 3)
        alpha=torch.transpose(alpha, 0, 2)
        alpha=alpha.unsqueeze(0)
        #alpha_b = 1 - alpha
        #alpha = torch.cat([alpha_b, alpha], 0)
        #print (image.size(),alpha.size())
        if self.phase == "train":

            sample = {'image': image, 'label': alpha, 'image_name': image_name}

        else:

            sample = {'image': image, 'label': alpha,  'image_name': label_img_path, 'alpha_shape': alpha.shape,'scale_list':scale_list,'pad_list':[min_yx,max_yx]}



        return sample


    def __len__(self):

        return len(self.fg)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #train_image_file = ImageFileTrain(alpha_dir=CONFIG.data.train_alpha)
    test_image_file = ImageFile("/home/ch229551/miccai2022/smalldiffuse/ANet/visualization/data_organize/HIE_split0.npy")
    #train_dataset = DataGenerator(train_image_file, phase='train')
    #test_image_file = ImageFiletest("/Users/baorina/Downloads/HIE_fold_split/HIE_split0.npy")

    test_dataset = DataGenerator(test_image_file, CONFIG,phase='test')
    #print ("ssssssss",len(test_dataset))
    countt=0
    for i in range(3,12):
        print (i)
        test_data=(test_dataset[i])
        #print (len(test_data))
        #print (test_data['fg'].shape,test_data['alpha'].shape)
        inputdata=test_data['image'].detach().cpu().numpy()

        gtmask=test_data['label'].detach().cpu().numpy()
        print ("inputdata, gtmask",inputdata.shape,gtmask.shape)
        for t in range(inputdata.shape[1]):
            adc=(inputdata[0,t,:,:]* 255).astype(np.uint8)
            zmap=(inputdata[1,t,:,:]* 255).astype(np.uint8)
            gtmask_=(gtmask[0,t,:,:]* 255).astype(np.uint8)
            showimg=np.concatenate([adc,zmap,gtmask_],1)
            #cv2.imshow("fig",showimg)
            #cv2.waitKey()
