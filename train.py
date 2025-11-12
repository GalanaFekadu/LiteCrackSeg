from data.dataset      import readIndex, dataReadPip, loadedDataset
from model.LiteCrackSeg import LiteCrackSeg
from trainer import DeepCrackTrainer
from config  import Config as cfg
from tqdm   import tqdm
import numpy as np
import torch, os, cv2, sys
import albumentations as A
import numpy as np
import cv2
import torch

IMAGENET_NORM = True

def get_train_aug_np():
    return A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.12),
        A.GaussNoise(var_limit=(10, 30), p=0.25),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.25),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=7,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ])


def get_val_aug_np():
    return A.Compose([])



_train_aug = get_train_aug_np()
_val_aug   = get_val_aug_np()

def albu_train_fn(img, mask):

    mask = ((mask > 0).astype(np.uint8) * 255)
    out = _train_aug(image=img, mask=mask)
    img_np  = out['image'].astype(np.uint8)     
    mask_np = out['mask'].astype(np.uint8)       
    return img_np, mask_np

def albu_val_fn(img, mask):
    mask = ((mask > 0).astype(np.uint8) * 255)
    out = _val_aug(image=img, mask=mask)
    img_np  = (out['image'] if 'image' in out else img).astype(np.uint8)
    mask_np = (out['mask']  if 'mask'  in out else mask).astype(np.uint8)
    return img_np, mask_np


def compute_global_ods_f1(prob_list, tgt_list, step=0.01):
    """ODS Dice/F1 over whole val-set."""
    probs    = torch.cat(prob_list).cpu().numpy().ravel()
    targets  = torch.cat(tgt_list).cpu().numpy().ravel()
    best_f1  = 0.0
    for t in np.arange(0.0, 1.0, step, dtype=np.float32):
        pred = probs > t
        tp   = np.sum(pred & (targets == 1))
        fp   = np.sum(pred & (targets == 0))
        fn   = np.sum(~pred & (targets == 1))
        if tp == 0:
            continue
        best_f1 = max(best_f1, 2 * tp / (2 * tp + fp + fn))
    return best_f1


def compute_global_ods_iou(prob_list, tgt_list, step=0.01):
    probs   = torch.cat(prob_list).cpu().numpy().ravel()
    targets = torch.cat(tgt_list).cpu().numpy().ravel()
    best_iou = 0.0
    for t in np.arange(0.0, 1.0, step, dtype=np.float32):
        pred = probs > t
        tp   = np.sum(pred & (targets == 1))
        fp   = np.sum(pred & (targets == 0))
        fn   = np.sum(~pred & (targets == 1))
        denom = tp + fp + fn
        if denom == 0:
            continue
        best_iou = max(best_iou, tp / denom)
    return best_iou



# training loop

def main():

    train_pipeline = dataReadPip(transforms=albu_train_fn)
    val_pipeline  = dataReadPip(transforms=None) 

    train_set = loadedDataset(readIndex(cfg.train_data_path, shuffle=True),
                              preprocess=train_pipeline)
    val_set   = loadedDataset(readIndex(cfg.val_data_path),
                              preprocess=val_pipeline)
    


    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size = cfg.train_batch_size,
        shuffle    = True,
        num_workers= 4,
        drop_last  = True
    )


    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size = cfg.val_batch_size,
        shuffle    = False,
        num_workers= 4,
        drop_last  = False    
    )

    steps_per_epoch = len(train_loader)


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device   = torch.device("cuda")
    num_gpu  = torch.cuda.device_count()

    model = LiteCrackSeg()

    model.to(device)          
    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    trainer = DeepCrackTrainer(model, steps_per_epoch=steps_per_epoch).to(device)

    FREEZE_ENCODER_EPOCHS = 0 

    encoder_to_freeze = model.module.encoder if num_gpu > 1 else model.encoder

    print(f"Freezing encoder for the first {FREEZE_ENCODER_EPOCHS} epochs.")
    for param in encoder_to_freeze.parameters():
        param.requires_grad = False


    #  load checkpoint

    if cfg.pretrained_model:
        pretrained = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model.load_state_dict({k: v for k, v in pretrained.items()
                               if k in model.state_dict()})
        trainer.vis.log(f'Loaded checkpoint {cfg.pretrained_model}', 'train info')


    #  Training hyper-parameters

    best_f1   = 0.0
    patience  = 25
    wait      = 0
    save_all  = getattr(cfg, 'save_all_ckpts', False) 


    #  Epoch loop

    for epoch in range(1, cfg.epoch + 1):            
        
        if epoch == FREEZE_ENCODER_EPOCHS + 1:
            print("* Unfreezing encoder weights. They will now be trained.")
            for param in encoder_to_freeze.parameters():
                param.requires_grad = True

        trainer.vis.log(f'Start Epoch {epoch}', 'train info')

        # Training 
        model.train()
        train_bar = tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         desc=f'Epoch {epoch} — Train',
                         leave=False)

        for idx, (img, lab) in train_bar:
            data   = img.cuda(non_blocking=True).float()
            target = lab.cuda(non_blocking=True).float()
            pred   = trainer.train_op(data, target)

            if idx % cfg.vis_train_loss_every == 0:
                trainer.vis.log(trainer.log_loss, 'train_loss')
                trainer.vis.plot_many({
                    'train_total_loss': trainer.log_loss['total_loss'],
                    'train_output_loss': trainer.log_loss['output_loss'],
                    'train_fuse5_loss':  trainer.log_loss['fuse5_loss'],
                    'train_fuse4_loss':  trainer.log_loss['fuse4_loss'],
                    'train_fuse3_loss':  trainer.log_loss['fuse3_loss'],
                    'train_fuse2_loss':  trainer.log_loss['fuse2_loss'],
                    'train_fuse1_loss':  trainer.log_loss['fuse1_loss'],
                })

            if idx % cfg.vis_train_acc_every == 0:
                trainer.acc_op(pred[0], target)
                trainer.vis.log(trainer.log_acc, 'train_acc')
                trainer.vis.plot_many({
                    'train_mask_acc':     trainer.log_acc['mask_acc'],
                    'train_mask_pos_acc': trainer.log_acc['mask_pos_acc'],
                    'train_mask_neg_acc': trainer.log_acc['mask_neg_acc'],
                })

            if idx % cfg.vis_train_img_every == 0:
                trainer.vis.img_many({
                    'train_img':   data.cpu(),
                    'train_output':torch.sigmoid(pred[0]).cpu(),
                    'train_lab':   target.unsqueeze(1).cpu(),
                    'train_fuse5': torch.sigmoid(pred[1]).cpu(),
                    'train_fuse4': torch.sigmoid(pred[2]).cpu(),
                    'train_fuse3': torch.sigmoid(pred[3]).cpu(),
                    'train_fuse2': torch.sigmoid(pred[4]).cpu(),
                    'train_fuse1': torch.sigmoid(pred[5]).cpu(),
                })

        # Validation
        model.eval()
        val_loss = dict.fromkeys([
            'eval_total_loss', 'eval_output_loss',
            'eval_fuse5_loss', 'eval_fuse4_loss',
            'eval_fuse3_loss', 'eval_fuse2_loss',
            'eval_fuse1_loss'
        ], 0.0)

        val_acc  = dict.fromkeys(['mask_acc','mask_pos_acc','mask_neg_acc'], 0.0)
        prob_cache, tgt_cache = [], []

        val_bar = tqdm(enumerate(val_loader),
                       total=len(val_loader),
                       desc=f'Epoch {epoch} — Val',
                       leave=False)

        with torch.no_grad():
            for _, (img, lab) in val_bar:
                v_data   = img.cuda(non_blocking=True).float()
                v_target = lab.cuda(non_blocking=True).float()

                v_pred = trainer.val_op(v_data, v_target)
                trainer.acc_op(v_pred[0], v_target)


                for k in ['total_loss','output_loss',
                          'fuse5_loss','fuse4_loss','fuse3_loss',
                          'fuse2_loss','fuse1_loss']:
                    val_loss[f'eval_{k}'] += trainer.log_loss[k]
                for k in ['mask_acc','mask_pos_acc','mask_neg_acc']:
                    val_acc[k] += trainer.log_acc[k]

                prob_cache.append(torch.sigmoid(v_pred[0]).cpu())
                tgt_cache .append(v_target.unsqueeze(1).cpu())


            trainer.vis.img_many({
                'eval_img':    v_data.cpu(),
                'eval_output': torch.sigmoid(v_pred[0]).cpu(),
                'eval_lab':    v_target.unsqueeze(1).cpu(),
                'eval_fuse5':  torch.sigmoid(v_pred[1]).cpu(),
                'eval_fuse4':  torch.sigmoid(v_pred[2]).cpu(),
                'eval_fuse3':  torch.sigmoid(v_pred[3]).cpu(),
                'eval_fuse2':  torch.sigmoid(v_pred[4]).cpu(),
                'eval_fuse1':  torch.sigmoid(v_pred[5]).cpu(),
            })

        num_val_batches = len(val_loader)

        trainer.vis.plot_many({k: v / num_val_batches for k, v in val_loss.items()})
        trainer.vis.plot_many({k: v / num_val_batches for k, v in val_acc .items()})

        epoch_f1  = compute_global_ods_f1 (prob_cache, tgt_cache)
        epoch_iou = compute_global_ods_iou(prob_cache, tgt_cache)
        trainer.vis.plot_many({'eval_ods_f1': epoch_f1, 'eval_ods_iou': epoch_iou})

 
  

        trainer.lr_sched.step(epoch)

        lr_info = {f'lr_group{i}': pg['lr'] for i, pg in enumerate(trainer.optimizer.param_groups)}
        trainer.vis.log(f'Epoch {epoch} LRs: ' + ', '.join([f'{k}={v:.2e}' for k,v in lr_info.items()]), 'train info')
        trainer.vis.plot_many(lr_info) 
        

        if epoch_f1 > best_f1 + 0.002:
            best_f1 = epoch_f1
            wait = 0
            tag = f'bestF1({epoch_f1:.4f})_IoU({epoch_iou:.4f})_epoch({epoch})'
            trainer.saver.save(model, tag=tag)
            trainer.vis.log(f'*** New best model ***: {tag}', 'train info')
        else:
            wait += 1
            trainer.vis.log(f'No improvement ({wait}/{patience})', 'train info')

        # save every epoch
        if save_all:
            trainer.saver.save(model, tag=f'{cfg.name}_epoch({epoch})')

        # Early stopping
        if wait >= patience:
            trainer.vis.log(f'Early stopping at epoch {epoch}', 'train info')
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[KeyboardInterrupt] Saving last state …')
        DeepCrackTrainer.saver.save(DeepCrackTrainer.model, tag='interrupt')
        sys.exit(0)
