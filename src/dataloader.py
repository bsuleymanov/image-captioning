import torch
import os, json
import numpy as np
import h5py
from pathlib import Path
from hydra.utils import to_absolute_path


class COCODataset:
    def __init__(self,
                 data_dir="datasets/coco_captioning",
                 max_len=None,
                 max_samples=None,
                 pca_features=True,
                 mode="train"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.max_len = max_len
        self.max_samples = max_samples
        self.pca_features = pca_features
        self.captions = None
        self.image_ids = None
        self.features = None
        self.idx_to_word = None
        self.word_to_idx = None
        self.urls = None
        self.load_coco_data()

    def load_coco_data(self):
        print(f'base dir {str(self.data_dir)}', )
        data = {}
        caption_file = self.data_dir / "coco2014_captions.h5"
        with h5py.File(caption_file, "r") as f:
            for k, v in f.items():
                if self.mode in k:
                    if "captions" in k and self.mode in k:
                        self.captions = np.asarray(v)
                    elif self.mode in k:
                        self.image_ids = np.asarray(v)

        if self.pca_features:
            feat_file = self.data_dir / f"{self.mode}2014_vgg16_fc7_pca.h5"
        else:
            feat_file = self.data_dir /  f"{self.mode}2014_vgg16_fc7.h5"
        with h5py.File(feat_file, "r") as f:
            self.features = np.asarray(f["features"])

        dict_file = self.data_dir / "coco2014_vocab.json"
        with open(dict_file, "r") as f:
            dict_data = json.load(f)
            for k, v in dict_data.items():
                if k == "idx_to_word":
                    self.idx_to_word = v
                else:
                    self.word_to_idx = v

        url_file = self.data_dir / f"{self.mode}2014_urls.txt"
        with open(url_file, "r") as f:
            self.urls = np.asarray([line.strip() for line in f])

        if self.max_len is not None:
            mask = np.random.randint(len(self), size=self.max_len)
            self.captions = self.captions[mask]
            self.image_ids = self.image_ids[mask]

    def __getitem__(self, idx):
        captions = self.captions[idx]
        image_ids = self.image_ids[idx]
        image_vecs = self.features[image_ids]
        urls = self.urls[image_ids]

        return captions, image_vecs, urls

    def __len__(self):
        if self.max_samples is None:
            return len(self.captions)
        return min(len(self.captions), self.max_samples)


class COCOLoader:
    def __init__(self, data_dir="datasets/coco_captioning",
                 max_len=None,
                 max_samples=None,
                 pca_features=True,
                 batch_size=8, mode="train",
                 drop_last=False, shuffle=True, num_workers=8):
        data_dir = Path(to_absolute_path(data_dir))
        dataset = COCODataset(data_dir=data_dir, max_len=max_len,
                              max_samples=max_samples,
                              pca_features=pca_features, mode=mode)
        self.loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            drop_last=drop_last
        )