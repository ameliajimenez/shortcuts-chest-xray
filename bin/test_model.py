import os
import sys
import argparse
import logging
import json
import time
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('--in_csv_path', default='dev.csv', metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--out_csv_path', default='test/test.csv',
                    metavar='OUT_CSV_PATH', type=str,
                    help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                                                               "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                                                                "comma separated, e.g. '0,1' ")

if not os.path.exists('test'):
    os.mkdir('test')


def get_pred(output, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()  # size: (batch_size,)
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return pred


def test_epoch(cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)
    y_score = []
    y_pred = []

    for step in range(steps):
        image, path = next(dataiter)
        image = image.to(device)
        output, __ = model(image)
        batch_size = len(path)
        pred = get_pred(output[0], cfg)
        pred_binary = (pred >= 0.5).astype(int)
        y_score.append(pred)
        y_pred.append(pred_binary)

    y_score_flat = [item for sublist in y_score for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]

    return y_score_flat, y_pred_flat


def run(args):
    with open(args.cfg_path + 'config.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt_path = os.path.join(args.model_path, 'best1.ckpt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.module.load_state_dict(ckpt['state_dict'])

    dataloader_test = DataLoader(
        ImageDataset(args.in_csv_path, cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    # load test data frame and save predictions
    df = pd.read_csv(args.in_csv_path)
    y_score, y_pred = test_epoch(cfg, args, model, dataloader_test)
    model_name = args.model_path.split('-')[1]
    data_seed = args.model_path.split('-')[2].split('/')[0]

    model_name_score = 'y_score_' + model_name + '_' + data_seed
    model_name_pred = 'y_pred_' + model_name + '_' + data_seed
    df[model_name_score] = pd.Series(y_score)
    df[model_name_pred] = pd.Series(y_pred)
    df.to_csv(os.path.join(args.cfg_path, 'my_test_with_preds.csv'), index=False)

    print('Save best is step :', ckpt['step'], 'AUC :', ckpt['auc_dev_best'])


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
