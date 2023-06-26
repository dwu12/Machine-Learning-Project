import os
from glob import glob 
import torch

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    HistogramNormalized,
    Resized,
    RandFlipd
)

from monai.handlers.utils import from_engine
from monai.networks.nets import UNet,DenseNet121
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ROCAUCMetric
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from monai.utils import first  

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pickle
#import pandas as pd


def main():
    f = open("cleaned_mgh_bch copy.csv", "r")

    pid = []
    target = []
    for i in f.readlines()[1:]:
        A = i.strip().split(',')
        name = A[0]
        outcome = int(A[1])
        pid.append(name)
        target.append(outcome)

    base_path = '/neuro/labs/grantlab/research/HIE_BCHNICU/2NIFTI_SS_REG/'
    path_list = []
    for i in pid:
        path = base_path+'{}/VISIT_01/reg_{}-VISIT_01-ADC_ss_in_MGH_ADC_Atlas_space.nii.gz'.format(i,i)
        path_list.append(path)   

    train_files = [{'img':img, "label":label} for img, label in zip(path_list[:50], target[:50])]

    val_files = [{'img':img, "label":label} for img, label in zip(path_list[50:60], target[50:60])]

    test_files = [{'img':img, "label":label} for img, label in zip(path_list[60:], target[60:])]

    train_transforms = Compose(
        [
            LoadImaged(keys = ['img']),
            EnsureChannelFirstd(keys = ['img']),
            ToTensord(keys = ['img','label'])
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys = ['img']),
            EnsureChannelFirstd(keys = ['img']),
            ToTensord(keys = ['img','label'])
        ]
    )

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot = 2)])

    train_dataset = Dataset(data = train_files, transform = train_transforms)
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle= True, num_workers = 4, pin_memory = torch.cuda.is_available())

    val_dataset = Dataset(data = val_files, transform = val_transforms)
    val_loader = DataLoader(val_dataset, batch_size = 4, num_workers = 4, pin_memory = torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(30):
        print("-" * 10)
        print(epoch + 1)
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print('train_loss = {}'.format(loss.item()))

        epoch_loss /= step 
        print('epoch {} average loss :{}'.format(epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_dict.pth")
                    print("saved new best metric model")
                    print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )

    print("train completed, best_metric: {} at epoch: {}".format(best_metric,best_metric_epoch))


if __name__ == '__main__':
    print('Is cuda availabel: ', torch.cuda.is_available())
    main()