import os
import cv2
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from utils.registry_class import DATASETS

@DATASETS.register_class()
class ImageDataset(Dataset):
    def __init__(self, 
            data_list, 
            data_dir_list,
            max_words=1000,
            vit_resolution=[224, 224],
            resolution=(384, 256),
            max_frames=1,
            transforms=None,
            vit_transforms=None,
            **kwargs):
        
        self.max_frames = max_frames
        self.resolution = resolution
        self.transforms = transforms
        self.vit_resolution = vit_resolution
        self.vit_transforms = vit_transforms

        image_list = []
        for item_path, data_dir in zip(data_list, data_dir_list):
            lines = open(item_path, 'r').readlines()
            lines = [[data_dir, item.strip()] for item in lines]
            image_list.extend(lines)
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        data_dir, file_path = self.image_list[index]
        img_key = file_path.split('|||')[0]
        try:
            ref_frame, vit_frame, video_data, caption = self._get_image_data(data_dir, file_path)
        except Exception as e:
            logging.info('{} get frames failed... with error: {}'.format(img_key, e))
            caption = ''
            img_key = ''
            ref_frame = torch.zeros(3, self.resolution[1], self.resolution[0])
            vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
            video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0]) 
        return ref_frame, vit_frame, video_data, caption, img_key

    def _get_image_data(self, data_dir, file_path):
        frame_list = []
        img_key, caption = file_path.split('|||')
        file_path = os.path.join(data_dir, img_key)
        for _ in range(5):
            try:
                image = Image.open(file_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                frame_list.append(image) 
                break
            except Exception as e:
                logging.info('{} read video frame failed with error: {}'.format(img_key, e))
                continue
        
        video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])
        try:
            if len(frame_list) > 0:
                mid_frame = frame_list[0]
                vit_frame = self.vit_transforms(mid_frame)
                frame_tensor = self.transforms(frame_list)
                video_data[:len(frame_list), ...] = frame_tensor
            else:
                vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        except:
            vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        ref_frame = copy(video_data[0])
        
        return ref_frame, vit_frame, video_data, caption

