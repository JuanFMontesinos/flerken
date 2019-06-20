#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .hmdb51 import HMDB51
from .video_jpg_ucf101_hmdb51 import class_process as _class_process_video2jpg
from .n_frames_ucf101_hmdb51 import class_process as _class_process_nframes
from .hmdb51_json import convert_hmdb51_csv_to_activitynet_json as _convert_hmdb51_csv_to_activitynet_json
from . import spatial_transforms,target_transforms,temporal_transforms
import os as _os
__all__ = ['video_jpg_ucf101_hmdb51','n_frames_ucf101_hmdb51','hmdb51_json','HMDB51']

def video_jpg_ucf101_hmdb51(avi_video_directory,jpg_video_directory):
    dir_path = avi_video_directory
    dst_dir_path = jpg_video_directory
    
    for class_name in _os.listdir(dir_path):
        _class_process_video2jpg(dir_path, dst_dir_path, class_name)    

def n_frames_ucf101_hmdb51(jpg_video_directory):
    dir_path = jpg_video_directory
    for class_name in _os.listdir(dir_path):
        _class_process_nframes(dir_path, class_name)
        
def hmdb51_json(annotation_dir_path):
    csv_dir_path = annotation_dir_path
    for split_index in range(1, 4):
        dst_json_path = _os.path.join(csv_dir_path, 'hmdb51_{}.json'.format(split_index))
        _convert_hmdb51_csv_to_activitynet_json(csv_dir_path, split_index, dst_json_path)