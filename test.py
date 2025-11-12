from data.dataset import readIndex, dataReadPip, loadedDataset

from model.LiteCrackSeg import LiteCrackSeg

from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test(test_data_path='datasets/TUT/test/test.txt', # path to dataset .txt file
         save_path='TUT_prediction_result', 
         pretrained_model='/home/Kaleb/LiteCrackSeg/checkpoints/TUT.pth', ): # path to pretrained model 
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    test_pipline = dataReadPip(transforms=None)

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)


    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = LiteCrackSeg()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model, steps_per_epoch=None).to(device) 

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()

    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
                device)
            test_pred = trainer.val_op(test_data, test_target)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())

            save_pred = test_pred.numpy() * 255

            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            cv2.imwrite(save_name, save_pred.astype(np.uint8))


if __name__ == '__main__':
    test()
