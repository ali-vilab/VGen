import os
import cv2
import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
from torch.utils.data import Dataset
from utils.registry_class import DATASETS


@DATASETS.register_class()
class VideoDataset(Dataset):
    def __init__(self, 
            data_list,
            data_dir_list,
            max_words=1000,
            resolution=(384, 256),
            vit_resolution=(224, 224),
            max_frames=16,
            sample_fps=8,
            transforms=None,
            vit_transforms=None,
            get_first_frame=False,
            **kwargs):

        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = resolution
        self.vit_resolution = vit_resolution
        self.sample_fps = sample_fps
        self.transforms = transforms
        self.vit_transforms = vit_transforms
        self.get_first_frame = get_first_frame
        
        image_list = []
        for item_path, data_dir in zip(data_list, data_dir_list):
            lines = open(item_path, 'r').readlines()
            lines = [[data_dir, item] for item in lines]
            image_list.extend(lines)
        self.image_list = image_list


    def __getitem__(self, index):
        data_dir, file_path = self.image_list[index]
        video_key = file_path.split('|||')[0]
        try:
            ref_frame, vit_frame, video_data, caption = self._get_video_data(data_dir, file_path)
        except Exception as e:
            logging.info('{} get frames failed... with error: {}'.format(video_key, e))
            caption = ''
            video_key = '' 
            ref_frame = torch.zeros(3, self.resolution[1], self.resolution[0])
            vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
            video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])        
        return ref_frame, vit_frame, video_data, caption, video_key
    
    
    def _get_video_data(self, data_dir, file_path):
        video_key, caption = file_path.split('|||')
        file_path = os.path.join(data_dir, video_key)

        for _ in range(5):
            try:
                capture = cv2.VideoCapture(file_path)
                _fps = capture.get(cv2.CAP_PROP_FPS)
                _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                stride = round(_fps / self.sample_fps)
                cover_frame_num = (stride * self.max_frames)
                if _total_frame_num < cover_frame_num + 5:
                    start_frame = 0
                    end_frame = _total_frame_num
                else:
                    start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                    end_frame = start_frame + cover_frame_num
                
                pointer, frame_list = 0, []
                while(True):
                    ret, frame = capture.read()
                    pointer +=1 
                    if (not ret) or (frame is None): break
                    if pointer < start_frame: continue
                    if pointer >= end_frame - 1: break
                    if (pointer - start_frame) % stride == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frame_list.append(frame)
                break
            except Exception as e:
                logging.info('{} read video frame failed with error: {}'.format(video_key, e))
                continue

        video_data = torch.zeros(self.max_frames, 3,  self.resolution[1], self.resolution[0])
        if self.get_first_frame:
            ref_idx = 0
        else:
            ref_idx = int(len(frame_list)/2)
        try:
            if len(frame_list)>0:
                mid_frame = copy(frame_list[ref_idx])
                vit_frame = self.vit_transforms(mid_frame)
                frames = self.transforms(frame_list)
                video_data[:len(frame_list), ...] = frames
            else:
                vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        except:
            vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        ref_frame = copy(frames[ref_idx])
        
        return ref_frame, vit_frame, video_data, caption

    def __len__(self):
        return len(self.image_list)


