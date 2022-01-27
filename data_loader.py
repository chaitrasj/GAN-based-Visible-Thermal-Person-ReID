from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import os
import itertools, random
from utils_ import *
from PIL import Image

def getGallerySet(data_path, shot, mode, ids):
    if mode == 'Indoor':
        rgb_cameras = ['cam1','cam2']
    else:
        rgb_cameras = ['cam1','cam2','cam4','cam5']
        
    data_path = data_path[:-15]
    
    files = []
    label = []
    cam_ = []
    for id in sorted(ids):
        random.shuffle(rgb_cameras)
        if shot == 'Single':
            for j in range(len(rgb_cameras)):
                camera = rgb_cameras[j]
                img_dir = os.path.join(data_path,camera,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    new = random.choice(new_files)
                    files.append(new)
                    cam_.extend([int(camera[-1])])
                    label.extend([int(id)])
                    
#                     break;
        else: 
            new_files = []
            for j in range(len(rgb_cameras)):
                camera = rgb_cameras[j]
                img_dir = os.path.join(data_path,camera,id)
                if os.path.isdir(img_dir):
                    multi = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    new_files.extend(multi)
            if new_files:
                new = random.sample(new_files, 10)
                files.extend(new)
                l,c = getLabels(new)
                cam_.extend(c)
                label.extend(l)
    return files, label, cam_, ids


def getFiles(data_path, ids, cameras, test=2):
    files = []
    label = []
    cam_ = []
    
    data_path = data_path[:-15]
    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
               ##########################
                if test!=2:
                    if test==1:
                        new_files = [new_files[0]]    #Test time of classifier
                    else:
                        del new_files[0]
                ##########################
                files.extend(new_files)
                cam_.extend([int(cam[-1])]*len(new_files))
                label.extend([int(id)]*len(new_files))
    return files, label, cam_


class SYSUDataTrain(Dataset):
    def __init__(self, data_path, FLAGS, transform):
        self.transform = transform
        self.FLAGS = FLAGS
        self.train_rgb_image, self.train_ir_image, self.train_rgb_label, self.train_ir_label, self.train_rgb_cam, self.train_ir_cam, self.id_train = getImageNames(data_path)
        self.serially_indexed_labels = np.array(getIds('../../SYSU-MM01/exp/available_id.txt'), dtype=int)
        
        print('Number of ids in Training data = ',len(self.id_train))
        createAllPermutations(self)
                
    def __getitem__(self, index):        
        X1 = self.transform(Image.open(self.rgb_list[index]['1']))
        X2 = self.transform(Image.open(self.rgb_list[index]['2']))
        X3 = self.transform(Image.open(self.rgb_list[index]['3']))

#         X1_ = self.transform(Image.open(self.ir_list[index]['1']))
#         X2_ = self.transform(Image.open(self.ir_list[index]['2']))
#         X3_ = self.transform(Image.open(self.ir_list[index]['3']))

        return {'1':X1, '2':X2, '3':X3}, self.rgb_list_label[index]
        
    def __len__(self):
        return len(self.rgb_list)

        
    
class SYSUDataTest(Dataset):
    def __init__(self, data_path, shot, mode, transform):

        # Define gallery and probe set for each mode and shot
        self.transform = transform
        
        if shot is not 'Single' and shot is not 'Multi':
            print('Give a valid shot, either Single or Multi')
        elif mode is not 'Indoor' and mode is not 'All':
            print('Give a valid mode, either Indoor or All')
        else:
            self.id = getIds(data_path)
            self.gall_names, self.gall_lab, self.gall_cam, self.id = getGallerySet(data_path, shot, mode, self.id)
            self.probe_names, self.probe_lab, self.probe_cam = getFiles(data_path, self.id, ['cam3','cam6'])    
            
    def __getitem__(self, index):
        if self.gallery:
            img1 = Image.open(self.gall_names[index])
            img1 = self.transform(img1)
            return img1, self.gall_lab[index], self.gall_cam[index]
        else:
            img1 = Image.open(self.probe_names[index])
            img1 = self.transform(img1)
            return img1, self.probe_lab[index], self.probe_cam[index]
            
    def __len__(self):
        if self.gallery:
            return len(self.gall_lab)
        else:
            return len(self.probe_lab)
    
    
class SYSUDataVal(Dataset):
    def __init__(self, data_path, FLAGS, transform,test=0):

        # Define gallery and probe set for each mode and shot
        self.transform = transform
        self.id_val = getIds(data_path)
        self.rgb_names, self.rgb_lab, self.rgb_cam = getFiles(data_path, self.id_val, ['cam1','cam2','cam4','cam5'],test)
        self.ir_names, self.ir_lab, self.ir_cam = getFiles(data_path, self.id_val, ['cam3','cam6'],test)
        self.id_val_int = np.array(self.id_val, dtype=int)
        print('Num of ids = ',len(self.id_val_int))
        self.FLAGS = FLAGS
        self.mode = FLAGS.mode #1 for RGB and 2 for IR
        
    def __getitem__(self, index):
        if self.mode==1:
            img_rgb = Image.open(self.rgb_names[index])
            img_rgb = self.transform(img_rgb)
            return img_rgb, self.rgb_lab[index]
        else: 
            img_ir = Image.open(self.ir_names[index])
            img_ir = self.transform(img_ir)
            return img_ir, self.ir_lab[index]

    def __len__(self):
        if self.mode==1:
            return len(self.rgb_lab)
        else:
            return len(self.ir_lab)

    
class RegDBTrain(Dataset):
    def __init__(self, data_path, train_color_list, train_thermal_list):
        
        self.train_rgb_image, self.train_rgb_label, self.train_ir_image, self.train_ir_label = getRegDB(data_path, train_color_list, train_thermal_list)
        
         # Init params
        self.train_ptr = 0
        # self.test_ptr = 0
        self.train_size = len(self.train_rgb_label)
        self.crop_size = 227
        self.scale_size = 256
        # self.mean = np.array([104., 117., 124.]) # original
        self.mean = np.array([123.68, 116.779, 103.939]) # ours
        self.n_classes = 206

        createAllPermutations(self)
        
    
    def __getitem__(self, index):
        return self.rgb_list[index], self.ir_list[index]

    def __len__(self):
        return len(self.rgb_list)
    

class RegDBTest(Dataset):
    def __init__(self, data_path, test_color_list, test_thermal_list):
        
        self.gall_names, self.gall_lab, self.probe_names, self.probe_lab = getRegDB(data_path, test_color_list, test_thermal_list)
        # Init params
        self.train_ptr = 0
        # self.test_ptr = 0
        self.crop_size = 227
        self.scale_size = 256
        # self.mean = np.array([104., 117., 124.]) # original
        self.mean = np.array([123.68, 116.779, 103.939]) # ours
        self.n_classes = 206
        
        
    def __getitem__(self, index):
        return self.probe_names[index]

    def __len__(self):
        return len(self.probe_lab)