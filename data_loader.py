import os
import cv2
import pydicom
import numpy as np

import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T


# data file extension
EXTENSION = ['.jpg', '.jpeg', '.png', '.tiff', '.dcm', '.dicom']


# data label
SIEMENS_label = {'B30f':0, 'B50f':1, 'B70f':2}
GE_label = {'CHEST':0}


# make dictionary list for image and label
def dict_list(images, mode, dataset):
    labels = [str(image.split('/')[-3]) for image in images]
    cluster, counts = np.unique(labels, return_counts=True)
    dict_list = [{'image':image, 'label':SIEMENS_label[label] if dataset=='SIEMENS' else GE_label[label]} for image, label in zip(images, labels)]
    print('# ====================================== #')
    if mode == 'train':
        print(f'Train-{dataset}')
        print(f'Train [Total] number = {len(images)}')
    else:
        print(f'Test-{dataset}')
        print(f'Test [Total] number = {len(images)}')
    print(f'Kernel Type List = {cluster}')
    print(f'Kernel Number List = {counts}')
    return dict_list


class Dataset(Dataset):
    def __init__(self, folder, mode, dataset, image_size=512):
        super().__init__()
        self.folder = os.path.join(folder, mode, dataset)
        self.image_size = image_size

        if os.path.isdir(self.folder):
            self.all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.folder) for root, _dirs, files in os.walk(self.folder) for fname in files}
        else:
            raise IOError('Path must point to a directory')
        
        self.image_fnames = sorted(fname for fname in self.all_fnames if self._file_ext(fname) in EXTENSION)
        if len(self.image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.dict_list = dict_list(self.image_fnames, mode, dataset)

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _open_file(self, fname):
        return open(os.path.join(self.folder, fname), 'rb')
    
    def _resize(self, img):
        if img.shape[0] != img.shape[1]:
            if img.shape[0] > img.shape[1]:
                padding = np.zeros((img.shape[0], (img.shape[0] - img.shape[1]) // 2))
                img = np.concatenate([padding, img, padding], 1)
            elif img.shape[0] < img.shape[1]:
                padding = np.zeros(((img.shape[1] - img.shape[0]) // 2), img.shape[1])
                img = np.concatenate([padding, img, padding], 0)
        else:
            if img.shape[0] < self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size), cv2.INTER_CUBIC)
            elif img.shape[0] > self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size), cv2.INTER_AREA)
        return img

    def _clip_and_normalize(self, img, min, max):
        img = np.clip(img, min, max)
        img = (img - min) / (max - min)
        return img
    
    def _CT_preprocess(self, dcm, img, window_width=None, window_level=None):
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        img = img * slope + intercept

        if window_width is not None and window_level is not None:
            min = window_level - (window_width / 2.0)
            max = window_level + (window_width / 2.0)
        else: # 12 bits
            min = -1024.0
            max = 3071.0

        img = self._clip_and_normalize(img, min, max)
        return img

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        fname = self.dict_list[index]['image']
        label = self.dict_list[index]['label']
        with self._open_file(fname) as f:
            path = f.name
            if self._file_ext(fname) == '.dcm' or '.dicom':
                dcm = pydicom.read_file(f, force=True)
                img = dcm.pixel_array.astype(np.float32)
                img = self._CT_preprocess(dcm, img)
            else: # jpg, jpeg, tiff, png, etc.
                img = np.array(Image.open(f))
        img = self._resize(img)
        dict_list = {'image':self.transform(img)*2 - 1, 'label':label, 'path':path}
        return dict_list


# get loader from Dataset as batch size
def get_loader(batch_size, root_path, dataset='SIEMENS', image_size=512, mode='train', shuffle=True, num_workers=4):
    dataset = Dataset(root_path, mode, dataset, image_size)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                             shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return dataloader