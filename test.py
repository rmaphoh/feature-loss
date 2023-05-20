# Copyright (c) Yukun Zhou 19/05/2023.
# All rights reserved.
# --------------------------------------------------------
import torch.nn.functional as F
import argparse
import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scripts.model import UNet
from scripts.dataset import LearningAVSegData
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from scipy.special import expit
from scripts.eval import eval_net
from skimage import filters
import pandas as pd
from scripts.utils import Define_image_size


def test_net(net_all, loader, device, mode, dataset_test):


    acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
        acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
            acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
                acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u  = eval_net(net_all, dataset_test, loader=loader, device=device, mode = mode, train_or='val')


    return acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
            acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
            acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
            acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u
    



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')
    parser.add_argument('--test_dataset', type=str, help='test dataset name', dest='test_data')
    parser.add_argument('--loss', type=str, help='loss name', dest='loss')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    img_size = Define_image_size(args.uniform, args.test_data)
    dataset_name = args.test_data
    checkpoint_saved = './checkpoints/{}/{}/CP_best.pth'.format(args.test_data,args.loss)
    csv_save = 'test_csv/{}/{}'.format(args.test_data,args.loss)

    if not os.path.isdir(csv_save):
        os.makedirs(csv_save)
    
    test_dir= "./data/{}/test/images/".format(dataset_name)
    test_label = "./data/{}/test/1st_manual/".format(dataset_name)
    test_mask =  "./data/{}/test/mask/".format(dataset_name)


    dataset = LearningAVSegData(test_dir, test_label, test_mask, img_size, dataset_name=dataset_name, train_or=False)
    test_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    net = UNet(input_channels=3, n_classes=4, bilinear=False)

    net.load_state_dict(torch.load(checkpoint_saved))
    net.eval()
    net.to(device=device)
    mode = 'whole'
    acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
    acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
    acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
    acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u = test_net(net_all=net, loader=test_loader, device=device, mode=mode, dataset_test=dataset_name)


Data4stage2 = pd.DataFrame({'ACC':[acc_a], 'Sensitivity':[sent_a], 'Specificity':[spet_a], 'Precision': [pret_a], 'G_value': [G_t_a], 'F1-score': [F1t_a], 'MSE': [mset_a], 'IOU': [iout_a], 'AUC-ROC': [auc_roct_a], 'AUC-PR': [auc_prt_a]})
Data4stage2.to_csv(csv_save+ '/results_artery.csv', index = None, encoding='utf8')

Data4stage2 = pd.DataFrame({'ACC':acc_v, 'Sensitivity':[sent_v], 'Specificity':[spet_v], 'Precision': [pret_v], 'G_value': [G_t_v], 'F1-score': [F1t_v], 'MSE': [mset_v], 'IOU': [iout_v], 'AUC-ROC': [auc_roct_v], 'AUC-PR': [auc_prt_v]})
Data4stage2.to_csv(csv_save+ '/results_vein.csv', index = None, encoding='utf8')

Data4stage2 = pd.DataFrame({'ACC':[acc], 'Sensitivity':[sent], 'Specificity':[spet], 'Precision': [pret], 'G_value': [G_t], 'F1-score': [F1t], 'MSE': [mset], 'IOU': [iout], 'AUC-ROC': [auc_roct], 'AUC-PR': [auc_prt]})

Data4stage2.to_csv(csv_save+ '/results_all.csv', index = None, encoding='utf8')
print('########################################3')
print('ARTERY')
print('#########################################')

print('Accuracy: ',  acc_a)
print('Sensitivity: ',  sent_a)
print('specificity: ',  spet_a)
print('precision: ',  pret_a)
print('G: ',  G_t_a)
print('F1_score_2: ',  F1t_a)
print('MSE: ',  mset_a)
print('iou: ',  iout_a)
print('auc_roc: ',  auc_roct_a)
print('auc_pr: ',  auc_prt_a)
print('auc_roc: ',   auc_roct_a)
print('auc_pr: ',   auc_prt_a)

#############################################3
print('########################################3')
print('VEIN')
print('#########################################')
#############################################3
print('Accuracy: ',  acc_v)
print('Sensitivity: ',  sent_v)
print('specificity: ',  spet_v)
print('precision: ',  pret_v)
print('G: ',  G_t_v)
print('F1_score_2: ',  F1t_v)
print('MSE: ',  mset_v)
print('iou: ',  iout_v)
print('auc_roc: ',  auc_roct_v)
print('auc_pr: ',  auc_prt_v)
print('auc_roc: ',   auc_roct_v)
print('auc_pr: ',   auc_prt_v)

###########################################
print('########################################3')
print('UNCERTAIN')
print('#########################################')
################################################
print('Accuracy: ',  acc_u)
print('Sensitivity: ',  sent_u)
print('specificity: ',  spet_u)
print('precision: ',  pret_u)
print('G: ',  G_t_u)
print('F1_score_2: ',  F1t_u)
print('MSE: ',  mset_u)
print('iou: ',  iout_u)
print('auc_roc: ',  auc_roct_u)
print('auc_pr: ',  auc_prt_u)
print('auc_roc: ',   auc_roct_u)
print('auc_pr: ',   auc_prt_u)

##########################################
print('########################################3')
print('AVERAGE')
print('#########################################')
##########################################
print('Accuracy: ',  acc)
print('Sensitivity: ',  sent)
print('specificity: ',  spet)
print('precision: ',  pret)
print('G: ',  G_t)
print('F1_score_2: ',  F1t)
print('MSE: ',  mset)
print('iou: ',  iout)
print('auc_roc: ',  auc_roct)
print('auc_pr: ',  auc_prt)
print('auc_roc: ',   auc_roct)
print('auc_pr: ',   auc_prt)


