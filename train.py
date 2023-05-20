# Copyright (c) Yukun Zhou 19/05/2023.
# All rights reserved.
# --------------------------------------------------------
import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pandas as pd
from scripts.pytorchtools import EarlyStopping
from scripts.eval import eval_net
from scripts.model import UNet
from scripts.utils import Define_image_size
from scripts.dataset import LearningAVSegData
import scripts.loss as loss_library
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              alpha = 0.5,
              beta = 1.1,
              gamma = 0.08,
              lr=0.001,
              val_percent=0.1,
              image_size=(592,880),
              save_cp=True,
              ):

    # define data path and checkpoint path
    dir_checkpoint="./checkpoints/{}/{}/".format(args.dataset,args.loss)
    train_dir= "./data/{}/training/images/".format(args.dataset)
    label_dir = "./data/{}/training/1st_manual/".format(args.dataset)
    mask_dir = "./data/{}/training/mask/".format(args.dataset)

    # create folders
    if not os.path.isdir(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    dataset = LearningAVSegData(train_dir, label_dir, mask_dir, image_size, args.dataset, train_or=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)


    writer = SummaryWriter(comment=f'Task_{args.dataset}_{args.loss}_LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')
    
    optimizer_ = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler_ = optim.lr_scheduler.ReduceLROnPlateau(optimizer_, 'min', factor=0.5, patience=50)
    early_stop_path = dir_checkpoint + 'es_checkpoint.pth'
    early_stop = EarlyStopping(patience=300,verbose=True, path=early_stop_path)
    
    L_seg_CE = nn.CrossEntropyLoss()
    
    if args.loss=='CF':
        loss = loss_library.CF_Loss(image_size,beta,alpha,gamma)
    
    elif args.loss=='cldice':
        loss = loss_library.soft_dice_cldice(iter_=20, alpha=0.2, smooth = 1.)
        # weight is from Table 1 in https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf
    elif args.loss=='GC':
        loss = loss_library.GC_2D(lmda=2.0)
        # weight is from Figure 3 in https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Zheng_Graph_Cuts_Loss_To_Boost_Model_Accuracy_and_Generalizability_for_ICCVW_2021_paper.pdf
    elif args.loss=='AC':
        loss = loss_library.active_contour_loss(weight=10.0) 
        # weight is from Figure 5 in https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf    
    else:
        raise Exception("Redefine the loss function")

    
    best_F1 = 0

    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['label']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                #encode_tensor = encode_tensor.to(device=device, dtype=torch.float32)

                optimizer_.zero_grad()
                masks_pred = net(imgs)

                G_Loss = loss(masks_pred, true_masks)
                
                writer.add_scalar('GLoss/G_train', G_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': G_Loss.item()})
                G_Loss.backward()
                optimizer_.step()


                pbar.update(imgs.shape[0])
                global_step += 1
                with torch.no_grad():
                    if global_step % (n_train // ( batch_size)) == 0:
                    #if True:
                        acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr, mse, iou,_ = eval_net(net, args.dataset, val_loader, device, mode='whole',train_or='train')[0:11]

                        scheduler_.step(G_Loss.item())
                        writer.add_scalar('learning_rate', optimizer_.param_groups[0]['lr'], global_step)
                        logging.info('Validation sensitivity: {}'.format(sensitivity))
                        writer.add_scalar('sensitivity/val_G', sensitivity, global_step)
                        logging.info('Validation specificity: {}'.format(specificity))
                        writer.add_scalar('specificity/val_G', specificity, global_step)
                        logging.info('Validation precision: {}'.format(precision))
                        writer.add_scalar('precision/val_G', precision, global_step)
                        logging.info('Validation G: {}'.format(G))
                        writer.add_scalar('G/val_G', G, global_step)
                        logging.info('Validation F1_score: {}'.format(F1_score_2))
                        writer.add_scalar('F1_score/val_G', F1_score_2, global_step)
                        logging.info('Validation mse: {}'.format(mse))
                        writer.add_scalar('mse/val_G', mse, global_step)
                        logging.info('Validation iou: {}'.format(iou))
                        writer.add_scalar('iou/val_G', iou, global_step)
                        logging.info('Validation acc: {}'.format(acc))
                        writer.add_scalar('Acc/val_G', acc, global_step)
                        logging.info('Validation auc_roc: {}'.format(auc_roc))
                        writer.add_scalar('Auc_roc/val_G', auc_roc, global_step)
                        logging.info('Validation auc_pr: {}'.format(auc_pr))
                        writer.add_scalar('Auc_pr/val_G', auc_pr, global_step)

                        early_stop(F1_score_2, net)  

                        if early_stop.early_stop:
                            print('Early stopping')
                            return         
            
            if F1_score_2 > best_F1:
                best_F1=F1_score_2
                if save_cp:
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net.state_dict(),
                            dir_checkpoint + f'CP_best.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=240, help='Number of epochs', dest='epochs')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--learning-rate', type=float, nargs='?', default=2e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file', dest='load')
    parser.add_argument('--dataset', type=str, help='dataset name', dest='dataset')
    parser.add_argument('--loss', type=str, help='loss name', dest='loss')
    parser.add_argument('--validation', type=float, default=5.0, help='Percent of the data validation', dest='val')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')
    parser.add_argument('--alpha', dest='alpha', type=float, help='alpha')
    parser.add_argument('--beta', dest='beta', type=float, help='beta')
    parser.add_argument('--gama', dest='gama', type=float, help='gama')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    seed_everything(38)
    img_size = Define_image_size(args.uniform, args.dataset)
    net = UNet(input_channels=3, n_classes=4, bilinear=False)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    train_net(net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gama,
                lr=args.lr,
                device=device,
                val_percent=args.val / 100,
                image_size=img_size)
    

