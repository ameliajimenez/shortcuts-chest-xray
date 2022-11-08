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


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--in_csv_path', default='dev.csv', metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                                                               "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                                                                "comma separated, e.g. '0,1' ")


def get_embedding(cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    embeddings = []
    for step in range(steps):
        image, path = next(dataiter)
        image = image.to(device)
        logits, logit_map, feat_map = model(image)
        embeddings.append(feat_map.cpu().detach().numpy())
    return embeddings


def run(args):
    with open(args.model_path + 'cfg.json') as f:
        cfg = edict(json.load(f))

    print(args.model_path + 'cfg.json')
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

    embeddings = get_embedding(cfg, args, model, dataloader_test)

    batch_size = 32
    new_embs = np.zeros([(len(embeddings)-1)*batch_size+embeddings[-1].shape[0], 1024])
    counter = 0
    for k in range(len(embeddings)):
        for bs in range(embeddings[k].shape[0]):
            new_embs[counter, ...] = embeddings[k][bs, :].squeeze()
            counter += 1

    # save embedding
    np.save('embedding_chexpert.npy', new_embs)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()