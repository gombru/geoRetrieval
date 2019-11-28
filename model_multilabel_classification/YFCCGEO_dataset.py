from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import image_processing
from PIL import Image
import json
import random

class YFCCGEO_dataset(Dataset):

    def __init__(self, root_dir, split, random_crop, mirror):

        self.split = split
        self.random_crop = random_crop
        self.mirror = mirror
        self.tags = []
        self.root_dir = root_dir

        print("Loading tag list ...")
        tags_file = self.root_dir + 'ordered_vocab.txt'
        for line in open(tags_file):
            self.tags.append(line.replace('\n',''))
        print("Vocabulary size: " + str(len(self.tags)))

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open(self.root_dir + 'splits/' + split))
        # self.num_elements = 96*4
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_names = []
        self.img_tags = np.zeros(self.num_elements, dtype=np.int64)

        # Read data
        print("Reading data ...")
        for i,line in enumerate(open(self.root_dir + 'splits/' + split)):
            if i == self.num_elements: break
            img_name = line.replace('\n','')
            self.img_names.append(img_name)
            tag_str = img_name.split('/')[0]
            tag_idx = self.tags.index(tag_str)
            self.img_tags[i] = tag_idx
            
        print("Data read. Set size: " + str(len(self.img_names)))


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = self.root_dir + 'img_resized/' + self.img_names[idx]

        # Load and transform img
        try:
            image = Image.open(img_name)
            if self.random_crop != 0:
                image = image_processing.RandomCrop(image,self.random_crop)
            if self.mirror:
                image = image_processing.Mirror(image)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)

            img_tag_idx = self.img_tags[idx]
        except:
            print("ERROR with image: " + str(img_name))
            print("Using default image")
            img_name = self.root_dir + 'img_resized/ski/3709131.jpg'
            image = Image.open(img_name)
            image = image_processing.RandomCrop(image,self.random_crop)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)
            img_tag_idx = self.tags.index('ski')

        # Get target vector (multilabel classification)
        target = np.zeros(100, dtype=np.float32)
        target[img_tag_idx] = 1
       
        # Build tensors
        img_tensor = torch.from_numpy(np.copy(im_np))
        target = torch.from_numpy(target)
        label = torch.from_numpy(np.array([img_tag_idx]))
        label = label.type(torch.LongTensor)

        return img_tensor, target, label