import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform

np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = {'1.0': '1', '1': '1', '': '0', '0.0': '0', '0': '0', '-1.0': '0', '-1': '0'}
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            idx_pneumothorax = header.index('Pneumothorax')
            self._label_header = [header[idx_pneumothorax]]
            for line in f:
                fields = line.strip('\n').split(',')

                labels = [int(self.dict.get(fields[idx_pneumothorax]))]
                self._labels.append(labels)

                image_path = fields[0]
                image_path = os.path.join(cfg.base_path, os.path.join(*(image_path.split(os.path.sep)[1:])))
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path

        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
