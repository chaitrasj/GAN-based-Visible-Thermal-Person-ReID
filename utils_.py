import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from itertools import cycle
from torch.utils.data import DataLoader
# from networks import Encoder_Dense, Decoder_Dense
from PIL import Image
import os, random, cv2
import pandas as pd
# from losses import TripletLoss
# from model import embed_net
import math
import itertools
import torch.nn.init as init


def getTransform(FLAGS):
#     normalize = transforms.Normalize(mean=[0.5],std=[0.5])
    transform_train = transforms.Compose([
#         transforms.ToPILImage(),
        transforms.Resize((FLAGS.image_height, FLAGS.image_width)),
        transforms.Pad(10),
        transforms.RandomCrop((FLAGS.image_height, FLAGS.image_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         normalize,
    ])
    transform_test = transforms.Compose([
#         transforms.ToPILImage(),
        transforms.Resize((FLAGS.image_height, FLAGS.image_width)),
        transforms.ToTensor(),
#         normalize,
    ])
    return transform_train, transform_test

transform_to_gray = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def weights_init_old(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
        
        
        
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

#################################### Data loader Helper functions ######################################

def getIds(data_path):
    with open(data_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_ = ["%04d" % x for x in ids]
        return id_
    

def getFiles(data_path, ids, cameras, test=2):
    files = []
    label = []
    cam_ = []
    
    a = data_path.split('/')[:-2]
    data_path = os.path.join(a[0],a[1],a[2])
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


def getGallerySet(data_path, shot, mode, ids):
    if mode == 'Indoor':
        rgb_cameras = ['cam1','cam2']
    else:
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    
    a = data_path.split('/')[:-2]
    data_path = os.path.join(a[0],a[1],a[2])
    
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
                    break;
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
    
def getImageNames(data_path):
    rgb_cameras = ['cam1','cam2','cam4','cam5']
    ir_cameras = ['cam3','cam6']

    id_train = getIds(data_path)
    
    files_rgb, label_rgb, cam_rgb = getFiles(data_path, id_train, rgb_cameras)
    files_ir, label_ir, cam_ir = getFiles(data_path, id_train, ir_cameras)
    return files_rgb, files_ir, label_rgb, label_ir, cam_rgb, cam_ir, np.array(id_train, dtype=int)


def createAllPermutations(self):

#     assert np.unique(self.train_rgb_label) == np.unique(self.train_ir_label), 'Number of Identities in Rgb and Ir data must be samse!'
    unique_labels = np.unique(self.train_rgb_label)
    rgb_list = []
    rgb_list_label = []
    ir_list = []
    ir_list_label = []
    
    for i in range(len(unique_labels)):
        id = unique_labels[i]

        # Creating triplets of V, T, T
        tmp_pos = [k for k,v in enumerate(self.train_rgb_label) if v==id]
        rgb = [self.train_rgb_image[i] for i in tmp_pos]
        
        tmp_pos = [k for k,v in enumerate(self.train_ir_label) if v==id]
        ir = [self.train_ir_image[i] for i in tmp_pos]
        
        M = min(len(rgb),len(ir))
        rgb_M = random.sample(rgb, M)
        ir_M_s = random.sample(ir, M)

        tmp_pos = [k for k,v in enumerate(self.train_ir_label) if v!=id]
        ir = [self.train_ir_image[i] for i in tmp_pos]
        ir_M_d = random.sample(ir, M)
        
        for i in range (M):
            dict = {}
            dict['1'] = rgb_M[i]
            dict['2'] = ir_M_s[i]
            dict['3'] = ir_M_d[i]
            rgb_list.append(dict)
            
            dict_lab = {}
            dict_lab['1'] = id
            dict_lab['2'] = id
            dict_lab['3'] = int(ir_M_d[i].split('/')[-2])
            rgb_list_label.append(dict_lab)


            
    # Creating triplets of T, V, V
        tmp_pos = [k for k,v in enumerate(self.train_ir_label) if v==id]
        ir = [self.train_ir_image[i] for i in tmp_pos]
        ir_M = random.sample(ir, M)

        tmp_pos = [k for k,v in enumerate(self.train_rgb_label) if v==id]
        rgb = [self.train_rgb_image[i] for i in tmp_pos]
        rgb_M_s = random.sample(rgb, M)

        tmp_pos = [k for k,v in enumerate(self.train_rgb_label) if v!=id]
        rgb = [self.train_rgb_image[i] for i in tmp_pos]
        rgb_M_d = random.sample(rgb, M)
        
        for i in range (M):
            dict = {}
            dict['1'] = ir_M[i]
            dict['2'] = rgb_M_s[i]
            dict['3'] = rgb_M_d[i]
            ir_list.append(dict)
            
            dict_lab = {}
            dict_lab['1'] = id
            dict_lab['2'] = id
            dict_lab['3'] = int(rgb_M_d[i].split('/')[-2])
            ir_list_label.append(dict_lab)
        
    self.rgb_list, self.rgb_list_label = rgb_list, rgb_list_label
    self.ir_list, self.ir_list_label = ir_list, ir_list_label
    return



def createAllPermutations_Single_Modality(self):

#     assert np.unique(self.train_rgb_label) == np.unique(self.train_ir_label), 'Number of Identities in Rgb and Ir data must be samse!'
    unique_labels = np.unique(self.train_rgb_label)
    rgb_list = []
    rgb_list_label = []
    
    for i in range(len(unique_labels)):
        id = unique_labels[i]

        # Creating triplets of V, V, V
        tmp_pos = [k for k,v in enumerate(self.train_rgb_label) if v==id]
        rgb = [self.train_rgb_image[i] for i in tmp_pos]
        
        M = 20
        combi = random.sample(list(itertools.combinations(rgb, 2)), M)
        
        
        tmp_pos = [k for k,v in enumerate(self.train_rgb_label) if v!=id]
        rgb = [self.train_rgb_image[i] for i in tmp_pos]
        combi_d = random.sample(rgb, M)
        
        for i in range (M):
            dict = {}
            dict['1'] = combi[i][0]
            dict['2'] = combi[i][1]
            dict['3'] = combi_d[i]
            rgb_list.append(dict)
            
            dict_lab = {}
            dict_lab['1'] = id
            dict_lab['2'] = id
            dict_lab['3'] = int(combi_d[i].split('/')[-2])
            rgb_list_label.append(dict_lab)

    self.rgb_list, self.rgb_list_label = rgb_list, rgb_list_label
    return



def getRegDB(data_path, color_list, thermal_list):
    name = data_path + color_list
    basePath = "/".join(data_path.split("/")[:-2]) + "/"
    with open(name) as f:
        data_color_list = open(name, 'rt').read().splitlines()
        # Get full list of color image and labels
        rgb_image = [basePath + s.split(' ')[0] for s in data_color_list]
        rgb_label = [int(s.split(' ')[1]) for s in data_color_list]

    name = data_path + thermal_list
    basePath = "/".join(data_path.split("/")[:-2]) + "/"
    with open(name) as f:
        data_thermal_list = open(name, 'rt').read().splitlines()
        # Get full list of thermal image and labels
        ir_image = [basePath + s.split(' ')[0] for s in data_thermal_list]
        ir_label = [int(s.split(' ')[1]) for s in data_thermal_list]

        return rgb_image, rgb_label, ir_image, ir_label



#################################### Testing Helper functions ##########################################


def get_state(path):
# original saved file with DataParallel
    state_dict = torch.load(path)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.
        new_state_dict[name] = v
    return new_state_dict

def getLabelIndex(ids):
    df = np.sort(np.array(ids))
    label_dict_gallery = {label:index for index, label in enumerate(df)}


def getLabels(probe_batch):
    label = []
    cam = []
    for i in range (len(probe_batch)):
        label.append(int(probe_batch[i].split('/')[-2]))
        cam.append(int(probe_batch[i].split('/')[-3][-1]))
    return label, cam


def getImage(minibatch, FLAGS):
    X = []
    for i in range (len(minibatch)):
        img = Image.open(minibatch[i])
        img = (img.resize((FLAGS.image_width, FLAGS.image_height), Image.ANTIALIAS))
        img = transform_test(img)
        X.append(img)
    X = torch.Tensor(np.stack(X))
    if FLAGS.cuda:
        X = X.cuda()
    return X


#################################### Evalutaion metric functions #######################################

# def getLoss(checkpoint, valset_sysu_mse, valset_sysu_trip, FLAGS, margin, transform_test):
#     loader_mse = cycle(DataLoader(valset_sysu_mse, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0, drop_last=True))
#     loader_trip = cycle(DataLoader(valset_sysu_trip, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0, drop_last=True))
        
#     net = embed_net(FLAGS.low_dim_bdtr, FLAGS.num_classes, drop = FLAGS.drop, arch=FLAGS.arch)
#     encoder_rgb = Encoder_Dense(FLAGS.embedding_dim, FLAGS.num_classes, FLAGS.feat_dim)
#     decoder_rgb = Decoder_Dense(FLAGS.embedding_dim, FLAGS.num_classes, FLAGS.feat_dim)
#     encoder_ir = Encoder_Dense(FLAGS.embedding_dim, FLAGS.num_classes, FLAGS.feat_dim)
#     decoder_ir = Decoder_Dense(FLAGS.embedding_dim, FLAGS.num_classes, FLAGS.feat_dim)
    
#     net.load_state_dict(checkpoint['net'])
#     encoder_ir.load_state_dict(checkpoint['state_dict_encoder_ir'])
#     encoder_rgb.load_state_dict(checkpoint['state_dict_encoder_rgb'])
#     decoder_ir.load_state_dict(checkpoint['state_dict_decoder_ir'])
#     decoder_rgb.load_state_dict(checkpoint['state_dict_decoder_rgb'])
    
#     net.eval()
#     encoder_rgb.eval()
#     encoder_ir.eval()
#     decoder_rgb.eval()
#     decoder_ir.eval()
    
#     encoder_rgb.cuda()
#     encoder_ir.cuda()
#     decoder_rgb.cuda()
#     decoder_ir.cuda()
#     net.cuda()
    
#     mse_loss = nn.MSELoss()
#     loss_mse = 0
# #     triplet_loss_fn = TripletLoss(margin)
# #     triplet_loss = 0
#     criterion = nn.CrossEntropyLoss()
#     identity_loss = 0
    
#     with torch.no_grad(): 
        
#         # MSE loss and Identity loss
#         for iteration in range (int(len(valset_sysu_mse) / FLAGS.batch_size)):
#             rgb, ir = next(loader_mse)
#             X_rgb = getImage(rgb, FLAGS)
#             X_ir = getImage(ir, FLAGS)
#             X_rgb, X_ir = net(X_rgb, X_ir)
            
#             S_, M_ = encoder_rgb(X_rgb)
#             recon_rgb = decoder_rgb(S_, M_)
#             loss_mse += mse_loss(X_rgb, recon_rgb).item()
            
            
#             S, M = encoder_ir(X_ir)
#             recon_ir = decoder_ir(S, M)
#             loss_mse += mse_loss(X_ir, recon_ir).item()
            
#             label_rgb = torch.LongTensor((np.nonzero(np.array(getLabels(rgb)[0])[:,None] == valset_sysu_mse.id_val_int)[1])).cuda()
#             label_ir = torch.LongTensor((np.nonzero(np.array(getLabels(ir)[0])[:,None] == valset_sysu_mse.id_val_int)[1])).cuda()
#             identity_loss += criterion(S_,label_rgb).item() + criterion(S,label_ir).item()
            
#         print('MSE loss on validation data = ',str(loss_mse))
        
#         print('Identity loss on validation data = ',str(identity_loss))
        
#         # Triplet loss
# #         for iteration in range (int(len(valset_sysu_trip) / FLAGS.batch_size)):
# #         num = 8
# #         for iteration in range (num):
# #             rgb_triplet, ir_triplet = next(loader_trip)

# #             S_1, _ = encoder_rgb(getImage(rgb_triplet['1'], FLAGS))
# #             S_2, _ = encoder_ir(getImage(rgb_triplet['2'], FLAGS))
# #             S_3, _ = encoder_ir(getImage(rgb_triplet['3'], FLAGS))
            
# #             S_11, _ = encoder_ir(getImage(ir_triplet['1'], FLAGS))
# #             S_22, _ = encoder_rgb(getImage(ir_triplet['2'], FLAGS))
# #             S_33, _ = encoder_rgb(getImage(ir_triplet['3'], FLAGS))
        
# #             triplet_loss += (triplet_loss_fn(S_1, S_2, S_3) + triplet_loss_fn(S_11, S_22, S_33)).item()
        
# #         print('Triplet loss on validation data = ',str(triplet_loss))
        
#         loss = loss_mse+identity_loss #+triplet_loss
#         print('Total Val loss = ',str(loss))
#     return loss_mse, identity_loss

# # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# # transform_test_sysu = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((args.img_h,args.img_w)),
# #     transforms.ToTensor(),
# #     normalize,
# # ])

# def mAPCheck(checkpoint, dataset, FLAGS, transform_test):
#             loader = cycle(DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0, drop_last=True))
#             net = embed_net(FLAGS.low_dim_bdtr, FLAGS.num_classes, drop = FLAGS.drop, arch=FLAGS.arch)
#             net.load_state_dict(checkpoint['net'])
#             net.eval()

#             with torch.no_grad(): 
#                 # Get the gallery features
#                 if FLAGS.data == 'sysu':
#                     X_gallery = torch.FloatTensor(len(dataset.gall_lab), FLAGS.num_channels, FLAGS.image_height, FLAGS.image_width)
#                     X_probe = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_height, FLAGS.image_width)
                    
#                     for i in range(len(dataset.gall_lab)):
#                         img = Image.open(dataset.gall_names[i])
#                         img = (img.resize((FLAGS.image_width, FLAGS.image_height), Image.ANTIALIAS))
#                         img = transform_test(img)
#                         X_gallery[i] = img
#                 else:
#                     scale_size = 256
#                     X_gallery = torch.FloatTensor(len(dataset.gall_lab), FLAGS.num_channels, scale_size, scale_size)
#                     X_probe = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, scale_size, scale_size)
                    
#                     for i in range(len(dataset.gall_lab)):
#                         img = Image.open(dataset.gall_names[i])
#                         img = (img.resize((scale_size, scale_size), Image.ANTIALIAS))
#                         img = transform_test(img)
#                         X_gallery[i] = img
                
#                 if FLAGS.cuda:
#                         X_gallery = X_gallery.cuda()
#                         net.cuda()

#                 X_gallery = net(X_gallery, X_gallery)[0]

# #             for iteration in range (math.ceil(len(dataset.gall_lab) / FLAGS.batch_size)):
# #                     S = net(Variable(X_gallery[iteration*FLAGS.batch_size : (iteration+1)*FLAGS.batch_size].cuda()),Variable(X_gallery[iteration*FLAGS.batch_size : (iteration+1)*FLAGS.batch_size].cuda()))[0]
# #                     if iteration==0:
# #                         S_gall = S
# #                     else:
# #                         S_gall = torch.cat((S_gall,S),0)
                    
# #                 S_gall = S_gall.view(S_gall.shape[0],-1).cuda()
#                 X_gallery = X_gallery.view(X_gallery.shape[0],-1).cuda()
#                 labels = []
#                 cameras = []
                
#                 # Getting probe features
#                 probe = []
#                 for iteration in range(int(len(dataset) / FLAGS.batch_size)):
#                         probe = next(loader)
                        
#                         if FLAGS.data == 'sysu':
#                             for i in range((FLAGS.batch_size)):
#                                 img = Image.open(probe[i])
#                                 img = (img.resize((FLAGS.image_width, FLAGS.image_height), Image.ANTIALIAS))
#                                 img = transform_test(img)
#                                 X_probe[i] = img
#                             label_probe, cam_probe = getLabels(probe)
#                             labels.extend(label_probe) 
#                             cameras.extend(cam_probe)

#                         else:
#                             for i in range((FLAGS.batch_size)):
#                                 img = Image.open(dataset.gall_names[i])
#                                 img = (img.resize((scale_size, scale_size), Image.ANTIALIAS))
#                                 img = transform_test(img)
#                                 X_probe[i] = img
                            
#                         X_probe_e = net(X_probe.cuda(), X_probe.cuda())[1]
# #                         S_probe, M_probe = encoder_ir(X_probe_e)
# #                         S_probe = S_probe.view(S_probe.shape[0],-1)
#                         X_probe_e = X_probe_e.view(X_probe_e.shape[0],-1)
#                         probe.append(X_probe_e)
# #                         if iteration==0:
# #                             A = torch.cdist(X_probe_e, X_gallery)
# #                         else:
# #                             A = torch.cat((A,torch.cdist(X_probe_e, X_gallery)), 0)
                            
#                 print('Number of gallery images = ')
#                 print('Number of query images = ')
#                 print(probe.shape)
#                 print(X_gallery.shape)
#                 A = torch.matmul(X_probe_e, X_gallery.transform(X_gallery.shape[1],X_gallery.shape[0]))

#                 print(A.shape)
                
#                 if FLAGS.data == 'sysu':
#                     cmc, mAP, mINP = eval_sysu(-A.cpu(), np.array(labels), np.array(dataset.gall_lab), np.array(cameras), np.array(dataset.gall_cam), max_rank = 20)
#                 else:
#                     cmc, mAP = eval_regdb(A.cpu(),  dataset.probe_lab, dataset.gall_lab, topk=20)
# #                     cmc, mAP, mINP = eval_sysu(A.cpu(), np.array(labels), np.array(dataset.gall_lab), np.array([1,2,3]), np.array([4,5,6]), max_rank = 20)
                    
#                 return mAP, cmc
                
                
# def getmAP(checkpoint, dataset, FLAGS, transform_test):
#             loader = cycle(DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0, drop_last=True))
#             net = embed_net(FLAGS.low_dim_bdtr, FLAGS.num_classes, drop = FLAGS.drop, arch=FLAGS.arch)
            
#             encoder_rgb = Encoder_Dense(FLAGS.embedding_dim, FLAGS.num_classes, FLAGS.feat_dim)
#             encoder_ir = Encoder_Dense(FLAGS.embedding_dim, FLAGS.num_classes, FLAGS.feat_dim)
            
#             encoder_ir.load_state_dict(checkpoint['state_dict_encoder_ir'])
#             encoder_rgb.load_state_dict(checkpoint['state_dict_encoder_rgb'])
#             net.load_state_dict(checkpoint['net'])
            
#             encoder_rgb.eval()
#             encoder_ir.eval()
#             net.eval()
            
#             with torch.no_grad(): 
#                 # Get the gallery features
#                 if FLAGS.data == 'sysu':
#                     X_gallery = torch.FloatTensor(len(dataset.gall_lab), FLAGS.num_channels, FLAGS.image_height, FLAGS.image_width)
#                     X_probe = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_height, FLAGS.image_width)
                    
#                     for i in range(len(dataset.gall_lab)):
#                         img = Image.open(dataset.gall_names[i])
#                         img = (img.resize((FLAGS.image_width, FLAGS.image_height), Image.ANTIALIAS))
#                         img = transform_test(img)
#                         X_gallery[i] = img
#                 else:
#                     scale_size = 256
#                     X_gallery = torch.FloatTensor(len(dataset.gall_lab), FLAGS.num_channels, scale_size, scale_size)
#                     X_probe = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, scale_size, scale_size)
                    
#                     for i in range(len(dataset.gall_lab)):
#                         img = Image.open(dataset.gall_names[i])
#                         img = (img.resize((scale_size, scale_size), Image.ANTIALIAS))
#                         img = transform_test(img)
#                         X_gallery[i] = img
                
#                 X_gallery_e = net(X_gallery, X_gallery)[0]
                
#                 if FLAGS.cuda:
#                     X_gallery.cuda()
#                     encoder_rgb.cuda()
#                     encoder_ir.cuda()
#                     net.cuda()
                
                
#                 for iteration in range (int(len(dataset.gall_lab) / FLAGS.batch_size)):
#                     S, M_gall = encoder_rgb(Variable(X_gallery_e[iteration*FLAGS.batch_size : (iteration+1)*FLAGS.batch_size].cuda()))
#                     if iteration==0:
#                         S_gall = S
#                     else:
#                         S_gall = torch.cat((S_gall,S),0)
                    
#                 S_gall = S_gall.view(S_gall.shape[0],-1).cuda()
#                 labels = []
#                 cameras = []
                
#                 # Getting probe features
                
#                 for iteration in range(int(len(dataset) / FLAGS.batch_size)):
#                         probe = next(loader)
                        
#                         if FLAGS.data == 'sysu':
#                             for i in range((FLAGS.batch_size)):
#                                 img = Image.open(probe[i])
#                                 img = (img.resize((FLAGS.image_width, FLAGS.image_height), Image.ANTIALIAS))
#                                 img = transform_test(img)
#                                 X_probe[i] = img
#                             label_probe, cam_probe = getLabels(probe)
#                             labels.extend(label_probe) 
#                             cameras.extend(cam_probe)

#                         else:
#                             for i in range((FLAGS.batch_size)):
#                                 img = Image.open(dataset.gall_names[i])
#                                 img = (img.resize((scale_size, scale_size), Image.ANTIALIAS))
#                                 img = transform_test(img)
#                                 X_probe[i] = img
                            
#                         X_probe_e = net(X_probe.cuda(), X_probe.cuda())[1]
#                         S_probe, M_probe = encoder_ir(X_probe_e)
#                         S_probe = S_probe.view(S_probe.shape[0],-1)
                        
#                         if iteration==0:
#                             A = torch.cdist(S_probe, S_gall)
#                         else:
#                             A = torch.cat((A,torch.cdist(S_probe, S_gall)), 0)
                
#                 if FLAGS.data == 'sysu':
#                     cmc, mAP, mINP = eval_sysu(A.cpu(), np.array(labels), np.array(dataset.gall_lab), np.array(cameras), np.array(dataset.gall_cam), max_rank = 20)
#                 else:
#                     cmc, mAP = eval_regdb(A.cpu(),  dataset.probe_lab, dataset.gall_lab, topk=20)
# #                     cmc, mAP, mINP = eval_sysu(A.cpu(), np.array(labels), np.array(dataset.gall_lab), np.array([1,2,3]), np.array([4,5,6]), max_rank = 20)
                    
#                 return mAP, cmc
            
def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    
    matches = np.zeros_like(g_pids[indices])
    matches[g_pids[indices] == q_pids[:, np.newaxis]] = 1#.astype(np.int32) # this m asy ybey not the right way
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        
        new_cmc = pred_label[q_idx][keep] # This is new_cmc after removing gall images for which, q images with cam as 3 which match with gall images having cam as 2
        new_index = np.unique(new_cmc, return_index=True)[1] # This is unique gall images after removing gall images in prvious step
        new_cmc = [new_cmc[index] for index in sorted(new_index)] # This is the new_cmc after removing the duplicate gall and only unique gall images are present
        
        new_match = np.zeros_like(new_cmc)
        new_match[new_cmc == q_pid] = 1 # This is new match array for this specific query image, which has 1 if its qid matches with any of the gid
        
        new_cmc = (new_match).cumsum() # This cumulative sum adds all 1s which match
        new_all_cmc.append(new_cmc[:max_rank]) # If max_rank is 20, then top 20 values of new_cmc is put in new_all_cmc
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank]) # This is 1 at position x, if in first x images,the qid is present, and len is max_rank
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
#     print('Length of all_cmc = ',len(all_cmc))
    all_cmc = np.array(all_cmc) #all_cmc is a list of arrays, each array is for each query image, and each array has length max_rank, with 1 at xth position, if till that 

    #xth position, any match is found (its the cumulative sums of all 1s basically)
#     print(all_cmc.shape)
#     print('num_valid_q = ',num_valid_q)
#     syss.exit()
    #,dtype=np.float32
    
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.array(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP


def eval_regdb (dist, query_ids, gallery_ids, topk = 20):
    single_gallery_shot=False
    first_match_break=True
    separate_camera_set=False
    distmat = dist
    m, n = distmat.shape
    # Fill up default values
    query_cams = np.zeros(m).astype(np.int32)
    gallery_cams = 2 * np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
#     matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    matches = np.zeros_like(gallery_ids[indices])
    matches[gallery_ids[indices] == query_ids[:, np.newaxis]] = True
    
    df = pd.DataFrame(gallery_ids[indices][20])
    df.to_csv('Gall_FirstQuery.csv', index=False)
#       print(query_ids[20])
#     df1 = pd.DataFrame(query_ids[0])
#     df1.to_csv('Query.cvs', index=False)

    
    # Compute AP for each query   
    ret = np.zeros(topk)
    num_valid_queries = 0
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |(gallery_cams[indices[i]] != query_cams[i]))
#         print(valid)
        if not np.any(matches[i, valid]): continue
        # Compute mAP
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        aps.append(average_precision_score(y_true, y_score))
        
        # Compute CMC  
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
            
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1

    mAP = np.mean(aps)
    cmc = ret.cumsum() / num_valid_queries
    return cmc, mAP

