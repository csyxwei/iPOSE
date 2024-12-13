import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 256
        opt.crop_size = 256
        opt.aspect_ratio = 1.0

        opt.label_nc = 3
        opt.contain_dontcare_label = False
        opt.semantic_nc = 2 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.labels, self.images, self.paths = self.list_images()

    def __len__(self,):
        return len(self.labels)

    def __getitem__(self, idx):
        label = Image.open(os.path.join(self.paths[0], self.labels[idx]))
        image = Image.open(os.path.join(self.paths[1], self.images[idx])).convert('RGB')
        label, image = self.transforms(label, image)
        label = label * 255
        return {"label": label, "image": image, "name": self.labels[idx]}

    def list_images(self):
        mode = "validation" if self.opt.phase == "test" or self.for_metrics else "training"
        labels = []
        images = []
        path_lab = self.opt.dataroot
        # path_lab = os.path.join(self.opt.dataroot, mode)
        cat = 'bed'
        for item in sorted(os.listdir(os.path.join(path_lab, mode, cat))):
            if item.find("label") != -1:
                labels.append(os.path.join(mode, cat, item))
            if item.find("image") != -1:
                images.append(os.path.join(mode, cat, item))


        assert len(labels)  == len(images), "different len of images and labels %s - %s" % (len(parts), len(labels))
        return labels, images, (path_lab, path_lab)

    def transforms(self, label, image):

        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return label, image