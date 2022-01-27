"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torchvision import datasets
import os, sys
import numpy as np
import random

class ReIDFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(ReIDFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)
        print(self.img_num)
        
    
    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('/')[-1][-5]
        try:
            cam_id = int(camera_id)
        except:
            print(filename,path,sys.exc_info()) 
            f = open("error.txt",'w')
            f.write(filename+","+path+","+str(sys.exc_info()))
            f.close()
            cam_id = 1
            
        return cam_id

    def _get_pos_sample(self, target, index, path):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        if len(pos_index)==0:  # in the query set, only one sample
            return path
        else:
            rand = random.randint(0,len(pos_index)-1)
        
         ############# ADDED BY CHAITRA #################
        cam_x = self._get_cam_id(path)
        xpos_path = self.samples[pos_index[rand]][0]
        cam_xpos = self._get_cam_id(xpos_path)

        dict = {1:0, 2:0, 4:0, 5:0, 3:1, 6:1}
        
        while(dict[cam_x] != dict[cam_xpos]):
            rand = random.randint(0,len(pos_index)-1)
            xpos_path = self.samples[pos_index[rand]][0]
            cam_xpos = self._get_cam_id(xpos_path)
            
        return self.samples[pos_index[rand]][0]
        
        
        ################################################
        
        return self.samples[pos_index[rand]][0]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        pos_path = self._get_pos_sample(target, index, path)
        pos = self.loader(pos_path)

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, pos

