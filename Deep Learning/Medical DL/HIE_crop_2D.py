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
from   torchvision import transforms

from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
CONFIG.data = EasyDict({})
CONFIG.data.real_world_aug = False
CONFIG.data.random_interp = True

CONFIG.data.crop_size=256
interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]




class ImageFileTrain():
    def __init__(self,data_file):
        super(ImageFileTrain, self).__init__()


        self.alpha=np.load(data_file,allow_pickle=True).item()['train']

    def __len__(self):
        return len(self.alpha)


class ImageFile():
    def __init__(self,data_file):
        super(ImageFile, self).__init__()


        self.alpha=np.load(data_file,allow_pickle=True).item()['val']

    def __len__(self):
        return len(self.alpha)

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
        image, alpha, = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        alpha = np.expand_dims(alpha, axis=0)

        sample['image'], sample['label']= torch.from_numpy(image), torch.from_numpy(alpha),


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



class DataGenerator(Dataset):
    def __init__(self, data, CONFIG,phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.model.crop_size
        self.fg = data.alpha
        self.pad=iaa.Sequential([
        iaa.CenterPadToFixedSize(width=self.crop_size , height=self.crop_size ,pad_mode='constant',pad_cval=0),
        ])
        self.resize = iaa.Sequential([
            iaa.Resize(self.crop_size)
        ])
        self.crop=iaa.Sequential([
        iaa.CenterCropToFixedSize(width=self.crop_size , height=self.crop_size )
        ])

        train_trans = [
            ToTensor(phase="train") ]

        test_trans = [ ToTensor() ]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]


    def __getitem__(self, idx):

        img_path, zmap_path, label_img_path, slice_num = self.fg[idx]

        slice_num = int(slice_num)

        img_adc_all = nib.load(img_path).get_data()
        img_z_all = nib.load(zmap_path).get_data()

        bou_xy = np.asarray(np.where(img_adc_all > 0))

        min_yx = np.min(bou_xy, 1)
        max_yx = np.max(bou_xy, 1)

        img_adc=img_adc_all[min_yx[0]:max_yx[0],min_yx[1]:max_yx[1],slice_num]
        img_z=img_z_all[min_yx[0]:max_yx[0],min_yx[1]:max_yx[1],slice_num]

        label_image = nib.load(label_img_path).get_data()[min_yx[0]:max_yx[0],min_yx[1]:max_yx[1], slice_num]

        label_image = (label_image==1)
        
        da_rina = np.concatenate([np.expand_dims(img_adc, -1), np.expand_dims(img_z, -1), np.expand_dims(img_z, -1),  np.expand_dims(label_image,-1)], -1)
        
        
        da_rina=np.asarray(da_rina,dtype=np.float32)

        da_rina=np.expand_dims(da_rina,0)

        da_rina = self.resize(images=da_rina)[0]

        da_rina_new = np.asarray(da_rina)

        image=da_rina_new[:,:,:2]
        alpha=1*(da_rina_new[:,:,-1]>0.5)
        alpha=np.asarray(alpha,dtype=np.float32)
        image_name = img_path

        if self.phase == "train":

            sample = {'image': image, 'label': alpha, 'image_name': image_name}

        else:

            sample = {'image': image, 'label': alpha,  'image_name': image_name, 'label_shape': alpha.shape}
        #print ("size check",image.shape,alpha.shape)
        sample = self.transform(sample)

        return sample


    def __len__(self):

        return len(self.fg)
        

import matplotlib.pyplot as plt


if __name__ == '__main__':

    #train_image_file = ImageFileTrain(alpha_dir=CONFIG.data.train_alpha)
    test_image_file = ImageFile("/home/ch229551/miccai2022/smalldiffuse/ANet/visualization/data_organize/HIE_split0.npy")
    #train_dataset = DataGenerator(train_image_file, phase='train')

    test_dataset = DataGenerator(test_image_file, phase='test')
    print ("length of the data",len(test_dataset))
    countt=0
    for i in range(len(test_dataset)):
        print (i)
        test_data=(test_dataset[i])
        #print (len(test_data))
        #print (test_data['fg'].shape,test_data['alpha'].shape)
        test_data['image']=test_data['image'].detach().cpu().numpy()
        test_data['label']=test_data['label'].detach().cpu().numpy()
      
        plt.imshow(np.concatenate([test_data['fg'][1,:,:],test_data['fg'][2,:,:],test_data['alpha'][0,:,:]],1))
        #plt.show()
        plt.savefig("test"+str(i)+".png",bbox_inches='tight')
    print (countt)