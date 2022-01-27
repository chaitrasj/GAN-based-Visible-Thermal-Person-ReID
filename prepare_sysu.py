import os
from shutil import copyfile
import argparse
from PIL import Image
import random

data_path = "./SYSU-MM01/"

parser = argparse.ArgumentParser(description='prepare')
opt = parser.parse_args()

if not os.path.isdir(data_path):
    print('please change the download_path')

save_path = "./SYSU/pytorch/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

###################################################
cameras = ['cam1','cam2','cam3','cam4','cam5','cam6']
rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
file_path_test   = os.path.join(data_path,'exp/test_id.txt')


# TRAIN ALL-TRAINING + VALIDATION
train_save_path = save_path + 'train_all'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)

with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    for x in ids:
        x = "%04d" % x
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,x)
            dst_path = os.path.join(train_save_path,x)

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            for root, dirs, files in os.walk(img_dir, topdown=True):
                for name in files:
                    ID = name.split('.')
                    copyfile(img_dir + '/' + name, dst_path + '/' + ID[0] + '_' + cam + '.jpg')
                    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    for x in ids:
        x = "%04d" % x
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,x)
            dst_path = os.path.join(train_save_path,x)

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            for root, dirs, files in os.walk(img_dir, topdown=True):
                for name in files:
                    ID = name.split('.')
                    copyfile(img_dir + '/' + name, dst_path + '/' + ID[0] + '_' + cam + '.jpg')


# PREPARING TEST DATA
test_save_path = save_path + 'query'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(file_path_test, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    for x in ids:
        x = "%04d" % x
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,x)
            dst_path = os.path.join(test_save_path,x)

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            for root, dirs, files in os.walk(img_dir, topdown=True):
                for name in files:
                    ID = name.split('.')
                    copyfile(img_dir + '/' + name, dst_path + '/' + ID[0] + '_' + cam + '.jpg')


# PREPARING Gallery DATA
test_save_path = save_path + 'gallery'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(file_path_test, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    for x in ids:
        x = "%04d" % x
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,x)
            dst_path = os.path.join(test_save_path,x)

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            for root, dirs, files in os.walk(img_dir, topdown=True):
                random.shuffle(files)
                for name in files:
                    ID = name.split('.')
                    copyfile(img_dir + '/' + name, dst_path + '/' + ID[0] + '_' + cam + '.jpg')
                    break;