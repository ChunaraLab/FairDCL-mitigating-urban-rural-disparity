# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import torch
import os
import cv2
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from torch.utils.data import SequentialSampler, RandomSampler

# class TwoCropsTransform:
#     """Take two random crops of one image as the query and key."""

#     def __init__(self, base_transform):
#         self.base_transform = base_transform

#     def __call__(self, x):
#         q = self.base_transform(x)
#         k = self.base_transform(x)
#         return [q, k]


class TwoCropsTransform(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []
        self.rgb_filename_list= []

        self.batch_generate(image_dir)

        self.transforms = transforms
        self.image_dir = image_dir

        urban = list(np.load('/scratch/mz2466/LoveDA_mixedD_moco/urban_ids.npy'))
        rural = list(np.load('/scratch/mz2466/LoveDA_mixedD_moco/rural_ids.npy'))

        self.joint_q1, self.joint_q2, self.joint_q3, self.joint_q4, self.joint_q = {}, {}, {}, {}, {}
        self.marginal = {}
        for name in urban:
            self.joint_q1[name] = torch.cat((torch.ones([1, 128, 128]), torch.zeros([1, 128, 128])), 0)
            self.joint_q2[name] = torch.cat((torch.ones([1, 64, 64]), torch.zeros([1, 64, 64])), 0)
            self.joint_q3[name] = torch.cat((torch.ones([1, 32, 32]), torch.zeros([1, 32, 32])), 0)
            self.joint_q4[name] = torch.cat((torch.ones([1, 16, 16]), torch.zeros([1, 16, 16])), 0)
            self.joint_q[name] = torch.cat((torch.ones([1, 1]), torch.zeros([1, 1])), 0).squeeze()
            '''
            self.marginal[name] = = []
            if (random.randint(0, 9) < 5):
                self.marginal[name].append(torch.cat((torch.ones([1, 128, 128]), torch.zeros([1, 128, 128])), 0))
            else:
                self.marginal[name].append(torch.cat((torch.zeros([1, 128, 128]), torch.ones([1, 128, 128])), 0))
            if (random.randint(0, 9) < 5):
                self.marginal[name].append(torch.cat((torch.ones([1, 64, 64]), torch.zeros([1, 64, 64])), 0))
            else:
                self.marginal[name].append(torch.cat((torch.zeros([1, 64, 64]), torch.ones([1, 64, 64])), 0))
            if (random.randint(0, 9) < 5):
                self.marginal[name].append(torch.cat((torch.ones([1, 32, 32]), torch.zeros([1, 32, 32])), 0))
            else:
                self.marginal[name].append(torch.cat((torch.zeros([1, 32, 32]), torch.ones([1, 32, 32])), 0))
            if (random.randint(0, 9) < 5):
                self.marginal[name].append(torch.cat((torch.ones([1, 16, 16]), torch.zeros([1, 16, 16])), 0))
            else:
                self.marginal[name].append(torch.cat((torch.zeros([1, 16, 16]), torch.ones([1, 16, 16])), 0))
            '''
        for name in rural:
            self.joint_q1[name] = torch.cat((torch.zeros([1, 128, 128]), torch.ones([1, 128, 128])), 0)
            self.joint_q2[name] = torch.cat((torch.zeros([1, 64, 64]), torch.ones([1, 64, 64])), 0)
            self.joint_q3[name] = torch.cat((torch.zeros([1, 32, 32]), torch.ones([1, 32, 32])), 0)
            self.joint_q4[name] = torch.cat((torch.zeros([1, 16, 16]), torch.ones([1, 16, 16])), 0)
            self.joint_q[name] = torch.cat((torch.zeros([1, 1]), torch.ones([1, 1])), 0).squeeze()
            '''
            self.rural_marginal[name] = = []
            if (random.randint(0, 9) < 5):
                self.rural_marginal.append(torch.cat((torch.ones([1, 128, 128]), torch.zeros([1, 128, 128])), 0))
            else:
                self.rural_marginal.append(torch.cat((torch.zeros([1, 128, 128]), torch.ones([1, 128, 128])), 0))
            if (random.randint(0, 9) < 5):
                self.rural_marginal.append(torch.cat((torch.ones([1, 64, 64]), torch.zeros([1, 64, 64])), 0))
            else:
                self.rural_marginal.append(torch.cat((torch.zeros([1, 64, 64]), torch.ones([1, 64, 64])), 0))
            if (random.randint(0, 9) < 5):
                self.rural_marginal.append(torch.cat((torch.ones([1, 32, 32]), torch.zeros([1, 32, 32])), 0))
            else:
                self.rural_marginal.append(torch.cat((torch.zeros([1, 32, 32]), torch.ones([1, 32, 32])), 0))
            if (random.randint(0, 9) < 5):
                self.rural_marginal.append(torch.cat((torch.ones([1, 16, 16]), torch.zeros([1, 16, 16])), 0))
            else:
                self.rural_marginal.append(torch.cat((torch.zeros([1, 16, 16]), torch.ones([1, 16, 16])), 0))
            '''

    def batch_generate(self, image_dir):
        # Change here !!!!
        #rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.npy'))
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))
        
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]

        self.rgb_filepath_list += rgb_filepath_list
        self.rgb_filename_list = rgb_filename_list

    def __getitem__(self, idx):
        # Change here !!!!
        #image = numpy.load(self.rgb_filepath_list[idx])
        #image = Image.fromarray(numpy.uint8(image)).convert('RGB')

        #img = cv2.imread(self.rgb_filepath_list[idx], -1)
        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        #image = Image.fromarray(img).convert("RGB")

    
        image = Image.open(self.rgb_filepath_list[idx]).convert("RGB")
        image_id = self.rgb_filepath_list[idx].split("/")[-1]
        
        #joint = torch.tensor()

        if self.transforms is not None:
            q = self.transforms(image)
            k = self.transforms(image)

        return [q, k], self.joint_q1[image_id], self.joint_q2[image_id], self.joint_q3[image_id], self.joint_q4[image_id], self.joint_q[image_id]
    

    def __len__(self):
        return len(self.rgb_filepath_list)




class TwoCropsTransform_med(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []
        self.rgb_filename_list= []

        self.batch_generate(image_dir)

        self.transforms = transforms
        self.image_dir = image_dir


    def batch_generate(self, image_dir):
        # Change here !!!!
        #rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.npy'))
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]

        self.rgb_filepath_list += rgb_filepath_list
        self.rgb_filename_list = rgb_filename_list

    def __getitem__(self, idx):
        # Change here !!!!
        #image = numpy.load(self.rgb_filepath_list[idx])
        #image = Image.fromarray(numpy.uint8(image)).convert('RGB')

        img = cv2.imread(self.rgb_filepath_list[idx], -1)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        image = Image.fromarray(img).convert("RGB")

        ## image = imread(self.rgb_filepath_list[idx])
        #image = Image.open(self.rgb_filepath_list[idx]).convert("RGB")
        image_id = self.rgb_filepath_list[idx].split("/")[-1]
        if self.transforms is not None:
            q = self.transforms(image)
            k = self.transforms(image)

        return [q, k], image_id

    def __len__(self):
        return len(self.rgb_filepath_list)



class TwoCropsTransform_rural(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []
        self.rgb_filename_list= []

        self.batch_generate(image_dir)

        self.transforms = transforms
        self.image_dir = image_dir


    def batch_generate(self, image_dir):
        # Change here !!!!
        #rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.npy'))
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]

        self.rgb_filepath_list += rgb_filepath_list
        self.rgb_filename_list = rgb_filename_list

    def __getitem__(self, idx):
        # Change here !!!!
        #image = numpy.load(self.rgb_filepath_list[idx])
        #image = Image.fromarray(numpy.uint8(image)).convert('RGB')

        #img = cv2.imread(self.rgb_filepath_list[idx], -1)
        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        #image = Image.fromarray(img).convert("RGB")

        ## image = imread(self.rgb_filepath_list[idx])
        image = Image.open(self.rgb_filepath_list[idx]).convert("RGB")
        image_id = self.rgb_filepath_list[idx].split("/")[-1]
        if self.transforms is not None:
            q = self.transforms(image)
            k = self.transforms(image)
        
        return [q, k], image_id

    def __len__(self):
        return len(self.rgb_filepath_list)



class TwoCropsTransform_urban(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []
        self.rgb_filename_list= []

        self.batch_generate(image_dir)

        self.transforms = transforms
        self.image_dir = image_dir


    def batch_generate(self, image_dir):
        # Change here !!!!
        #rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.npy'))
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]

        self.rgb_filepath_list += rgb_filepath_list
        self.rgb_filename_list = rgb_filename_list

    def __getitem__(self, idx):
        # Change here !!!!
        #image = numpy.load(self.rgb_filepath_list[idx])
        #image = Image.fromarray(numpy.uint8(image)).convert('RGB')

        ## image = imread(self.rgb_filepath_list[idx])
        #image = Image.open(self.rgb_filepath_list[idx]).convert("RGB")
        
        #img = cv2.imread(self.rgb_filepath_list[idx], -1)
        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        #image = Image.fromarray(img).convert("RGB")
        
        image = Image.open(self.rgb_filepath_list[idx]).convert("RGB")
        image_id = self.rgb_filepath_list[idx].split("/")[-1]
        if self.transforms is not None:
            q = self.transforms(image)
            k = self.transforms(image)

        return [q, k], image_id

    def __len__(self):
        return len(self.rgb_filepath_list)





class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
