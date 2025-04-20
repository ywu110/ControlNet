import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np


class MnistDataset(Dataset):

    def __init__(self, split, im_path, im_ext='png', use_condition=True):

        self.split = split
        self.im_ext = im_ext
        self.images = self.load_images(im_path)
        self.use_condition = use_condition
        
    def load_images(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        
        if self.use_condition:
            canny_image = Image.open(self.images[index])
            canny_image = np.array(canny_image)
            canny_image = cv2.Canny(canny_image, 100, 200)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)
            return im_tensor, canny_image_tensor
        else:
            return im_tensor
