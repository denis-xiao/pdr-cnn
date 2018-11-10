from torch.utils.data import Dataset
from torchvision import transforms as T
from config import config
from PIL import Image
from itertools import chain
from glob import glob
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import os
import cv2
import torch


def get_files(root,mode):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test":
        #for train and val
        all_data_path,labels = [],[]
        image_folders = list(map(lambda x:root+x,os.listdir(root)))
        jpg_image_1 = list(map(lambda x:glob(x+"/*.jpg"),image_folders))
        jpg_image_2 = list(map(lambda x:glob(x+"/*.JPG"),image_folders))
        all_images = list(chain.from_iterable(jpg_image_1 + jpg_image_2))
        print("loading train dataset")
        for file in tqdm (all_images):
            all_data_path.append(file)
            labels.append((int(file.split("\\")[-2])))
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        print(all_data_path)
        print(labels)
        return all_files
    else:
        print("check the mode please!")

train_ = get_files("D:/document/python/plants_disease_detection-master/plants_disease_detection-master/data/train/", "train")
