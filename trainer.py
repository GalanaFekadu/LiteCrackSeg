from torch import nn
from tools.visdom import Visualizer
from tools.checkpointer import Checkpointer
from config import Config as cfg
import torch
import torch.nn.functional as F
from losses import BCELoss, DiceLoss, FocalLoss, TverskyLoss


def get_optimizer(model):
    if cfg.use_adam:
        return torch.optim.AdamW(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay,
                                 betas=(0.9, 0.999))
    else:
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, )

class DeepCrackTrainer(nn.Module):
    def __init__(self, model, steps_per_epoch=None):
        super().__init__()
        self.vis = Visualizer(env=cfg.vis_env)
        self.model = model

        self.saver = Checkpointer(cfg.name, cfg.saver_path, overwrite=False, verbose=True, timestamp=True,
                                  max_queue=cfg.max_save)

        self.optimizer = get_optimizer(self.model)

        
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0= 10, 
            T_mult= 2,  
            eta_min=1e-6 
        )

        self.criterion = TverskyLoss()


        self.log_loss = {}
        self.log_acc = {}


    def train_op(self, input, target):
        self.optimizer.zero_grad()

        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1 = self.model(input)

        output_loss = self.criterion(pred_output, target)
        fuse5_loss  = self.criterion(pred_fuse5,  target)
        fuse4_loss  = self.criterion(pred_fuse4,  target)
        fuse3_loss  = self.criterion(pred_fuse3,  target)
        fuse2_loss  = self.criterion(pred_fuse2,  target)
        fuse1_loss  = self.criterion(pred_fuse1,  target)

        total_loss=output_loss


        total_loss.backward()
        self.optimizer.step()


        self.log_loss = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss': fuse5_loss.item(),
            'fuse4_loss': fuse4_loss.item(),
            'fuse3_loss': fuse3_loss.item(),
            'fuse2_loss': fuse2_loss.item(),
            'fuse1_loss': fuse1_loss.item()
        }

        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,

    def val_op(self, input, target):
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1, = self.model(input)

        output_loss = self.criterion(pred_output, target)
        fuse5_loss  = self.criterion(pred_fuse5,  target)
        fuse4_loss  = self.criterion(pred_fuse4,  target)
        fuse3_loss  = self.criterion(pred_fuse3,  target)
        fuse2_loss  = self.criterion(pred_fuse2,  target)
        fuse1_loss  = self.criterion(pred_fuse1,  target)
        
        total_loss=output_loss
    
        self.log_loss = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss': fuse5_loss.item(),
            'fuse4_loss': fuse4_loss.item(),
            'fuse3_loss': fuse3_loss.item(),
            'fuse2_loss': fuse2_loss.item(),
            'fuse1_loss': fuse1_loss.item()
        }

        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,


    def acc_op(self, pred, target):

        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

   
        gt = (target > 0.5).float()

        prob     = torch.sigmoid(pred)
        bin_pred = (prob > cfg.acc_sigmoid_th).float()

        gt2d   = gt[:, 0]      
        pr2d   = bin_pred[:, 0]

        total   = gt2d.numel()
        pos_cnt = int(gt2d.sum().item())
        neg_cnt = total - pos_cnt


        mask_acc = (pr2d == gt2d).float().mean().item()


        mask_pos_acc = (pr2d[gt2d == 1] == 1).float().mean().item() if pos_cnt else 0.0
        mask_neg_acc = (pr2d[gt2d == 0] == 0).float().mean().item() if neg_cnt else 0.0

        self.log_acc = {
            'mask_acc': mask_acc,
            'mask_pos_acc': mask_pos_acc,
            'mask_neg_acc': mask_neg_acc,
        }
