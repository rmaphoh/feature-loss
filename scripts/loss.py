# Copyright (c) Yukun Zhou 19/05/2023.
# All rights reserved.
# Code of AC is from https://github.com/xuuuuuuchen/Active-Contour-Loss
# Code of GC is from https://github.com/zzhenggit/graph_cuts_loss
# Code of Cldice is from https://github.com/jocpae/clDice
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import soft_skel


def encode_mask(ground_truth,prediction):
    encode_tensor=F.one_hot(ground_truth.to(torch.int64), num_classes=4)
    encode_tensor=encode_tensor.permute(0, 3, 1, 2).contiguous()     
    encode_tensor = encode_tensor.to(device=torch.device('cuda'), dtype=torch.float32)
    masks_pred_softmax = F.softmax(prediction,dim=1)
    
    return encode_tensor,masks_pred_softmax



class CF_Loss(nn.Module):

    def __init__(self, img_size,beta,alpha,gamma):
        super(CF_Loss, self).__init__()

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.p = torch.tensor(img_size[-1], dtype = torch.float)
        self.n = torch.log(self.p)/torch.log(torch.tensor([2]).to('cuda'))
        self.n = torch.floor(self.n)
        self.sizes = 2**torch.arange(self.n.item(), 1, -1).to(dtype=torch.int)
        self.CE = nn.CrossEntropyLoss()
    
    def get_count(self,sizes,p,masks_pred_softmax):
    
        counts = torch.zeros((masks_pred_softmax.shape[0], len(sizes),2))
        index = 0

        for size in sizes:

            stride = (size, size)
            pad_size = torch.where((p%size) == 0, torch.tensor(0,dtype = torch.int), (size - p%size).to(dtype=torch.int))
            pad = nn.ZeroPad2d((0,pad_size, 0, pad_size))
            pool = nn.AvgPool2d(kernel_size = (size, size), stride = stride)

            S = pad(masks_pred_softmax)
            S = pool(S)
            S = S*((S> 0) & (S < (size*size)))
            counts[...,index,0] = (S[:,0,...] - S[:,2,...]).abs().sum()/(S[:,2,...]>0).sum()
            counts[...,index,1] = (S[:,1,...] - S[:,3,...]).abs().sum()/(S[:,3,...]>0).sum()        

            index += 1

        return counts

    def forward(self, prediction, ground_truth):
        

        encode_tensor,masks_pred_softmax = encode_mask(ground_truth,prediction)
        
        loss_CE = self.CE(prediction, ground_truth)
        
        Loss_vd = (torch.abs(masks_pred_softmax[:,1,...].sum()-encode_tensor[:,1,...].sum())+torch.abs(masks_pred_softmax[:,2,...].sum()-encode_tensor[:,2,...].sum()))/(masks_pred_softmax.shape[0]*masks_pred_softmax.shape[2]*masks_pred_softmax.shape[3])
        
        masks_pred_softmax = masks_pred_softmax[:,1:3,...]
        encode_tensor = encode_tensor[:,1:3,...]
        masks_pred_softmax = torch.cat((masks_pred_softmax, encode_tensor), 1)
        counts = self.get_count(self.sizes,self.p,masks_pred_softmax)

        artery_ = torch.sqrt(torch.sum(self.sizes*((counts[...,0])**2)))
        vein_ = torch.sqrt(torch.sum(self.sizes*((counts[...,1])**2)))
        size_t = torch.sqrt(torch.sum(self.sizes**2))
        loss_FD = (artery_+vein_)/size_t/masks_pred_softmax.shape[0]
        
        loss_value = self.beta*loss_CE + self.alpha*loss_FD + self.gamma*Loss_vd
        
        return loss_value



# original 2D GC loss with no approximation
class GC_2D_Original(torch.nn.Module):

    def __init__(self, lmda, sigma):
        super(GC_2D_Original, self).__init__()
        self.lmda = lmda
        self.sigma = sigma

    def forward(self, input, target):
        # input: B * C * H * W, after sigmoid operation
        # target: B * C * H * W

        target,input = encode_mask(target,input)
        # region term equals to BCE
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary_term
        '''
        x5 x1 x6
        x2 x  x4
        x7 x3 x8
        '''
        # vertical: x <-> x1, x3 <-> x1
        target_vert = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # delta(yu, yv)
        input_vert = input[:, :, 1:, :] - input[:, :, :-1, :]  # pu - pv

        # horizontal: x <-> x2, x4 <-> x
        target_hori = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # delta(yu, yv)
        input_hori = input[:, :, :, 1:] - input[:, :, :, :-1]  # pu - pv

        # diagonal1: x <-> x5, x8 <-> x
        target_diag1 = torch.abs(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])  # delta(yu, yv)
        input_diag1 = input[:, :, 1:, 1:] - input[:, :, :-1, :-1]  # pu - pv

        # diagonal2: x <-> x7, x6 <-> x
        target_diag2 = torch.abs(target[:, :, 1:, :-1] - target[:, :, :-1, 1:])  # delta(yu, yv)
        input_diag2 = input[:, :, 1:, :-1] - input[:, :, :-1, 1:]  # pu - pv

        dist1 = 1.0  # dist(u, v), e.g. x <-> x1
        dist2 = 2.0 ** 0.5  # dist(u, v) , e.g. x <-> x6

        p1 = torch.exp(-(input_vert ** 2) / (2 * self.sigma * self.sigma)) / dist1 * target_vert
        p2 = torch.exp(-(input_hori ** 2) / (2 * self.sigma * self.sigma)) / dist1 * target_hori

        p3 = torch.exp(-(input_diag1 ** 2) / (2 * self.sigma * self.sigma)) / dist2 * target_diag1
        p4 = torch.exp(-(input_diag2 ** 2) / (2 * self.sigma * self.sigma)) / dist2 * target_diag2

        boundary_term = (torch.sum(p1) / torch.sum(target_vert) +
                         torch.sum(p2) / torch.sum(target_hori) +
                         torch.sum(p3) / torch.sum(target_diag1) +
                         torch.sum(p4) / torch.sum(target_diag2)) / 4  # equation (5)

        return self.lmda * region_term + boundary_term


# 2D GC loss with boundary approximation in equation (7) to eliminate sigma
class GC_2D(torch.nn.Module):

    def __init__(self, lmda):
        super(GC_2D, self).__init__()
        self.lmda = lmda

    def forward(self, input, target):
        # input: B * C * H * W, after sigmoid operation
        # target: B * C * H * W

        target,input = encode_mask(target,input)
        # region term equals to BCE
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary_term
        '''
        x5 x1 x6
        x2 x  x4
        x7 x3 x8
        '''
        # vertical: x <-> x1, x3 <-> x1
        target_vert = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # delta(yu, yv)
        input_vert = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])  # |pu - pv|

        # horizontal: x <-> x2, x4 <-> x
        target_hori = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # delta(yu, yv)
        input_hori = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])  # |pu - pv|

        # diagonal1: x <-> x5, x8 <-> x
        target_diag1 = torch.abs(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])  # delta(yu, yv)
        input_diag1 = torch.abs(input[:, :, 1:, 1:] - input[:, :, :-1, :-1])  # |pu - pv|

        # diagonal2: x <-> x7, x6 <-> x
        target_diag2 = torch.abs(target[:, :, 1:, :-1] - target[:, :, :-1, 1:])  # delta(yu, yv)
        input_diag2 = torch.abs(input[:, :, 1:, :-1] - input[:, :, :-1, 1:])  # |pu - pv|

        p1 = input_vert * target_vert
        p2 = input_hori * target_hori
        p3 = input_diag1 * target_diag1
        p4 = input_diag2 * target_diag2

        boundary_term = 1 - (torch.sum(p1) / torch.sum(target_vert) +
                             torch.sum(p2) / torch.sum(target_hori) +
                             torch.sum(p3) / torch.sum(target_diag1) +
                             torch.sum(p4) / torch.sum(target_diag2)) / 4  # equation (7), and normalized to (0,1)

        return self.lmda * region_term + boundary_term


    
    
class active_contour_loss(torch.nn.Module):
    def __init__(self, weight):
        super(active_contour_loss, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true):
        # input: B * C * H * W * D, after sigmoid operation
        # target: B * C * H * W * D

        y_true,y_pred = encode_mask(y_true,y_pred)
        
        '''
        y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
        delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
        delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c) 

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
        # region term
        c_in  = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)
        region_in  = torch.mean(y_pred * (y_true - c_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1-y_pred)* (y_true - c_out)**2 ) 
        region = region_in + region_out
        loss =  lenth + self.weight*region
        
        return loss


    
    
    
class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+smooth)/(torch.sum(skel_pred[:,1:,...])+smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+smooth)/(torch.sum(skel_true[:,1:,...])+smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        
        y_true,y_pred = encode_mask(y_true,y_pred)
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
