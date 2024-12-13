import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import collections
class PartSplitDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        # opt.load_size = 144
        # opt.crop_size = 128
        opt.load_size = 72
        opt.crop_size = 64
        opt.label_nc = 3
        opt.contain_dontcare_label = False
        opt.semantic_nc = 9 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        opt.categorys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 36, 37]  # 35 is ADE car, 36 is ADE washer

        if not for_metrics:
            opt.selected_cats = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28]
        else:
            opt.selected_cats = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28]

        self.opt = opt
        self.for_metrics = for_metrics
        self.labels, self.parts, self.colors, self.cats, self.paths = self.list_images()
        self.cat_count = dict(collections.Counter(self.cats))


    def __len__(self,):
        return len(self.labels)

    def __getitem__(self, idx):
        label = Image.open(os.path.join(self.paths[0], self.labels[idx])).convert('RGB')
        part = Image.open(os.path.join(self.paths[1], self.parts[idx]))
        color = Image.open(os.path.join(self.paths[2], self.colors[idx])).convert('RGB')
        cat = torch.tensor(self.cats[idx])
        w = 1.0 * len(self.cats) / self.cat_count[self.cats[idx]]
        w = torch.tensor(w)
        label, part, color = self.transforms(label, part, color)
        label = label
        part = part * 255
        return {"label": label, "part": part, "name": self.labels[idx], 'cat':cat, 'color':color, 'w':w}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        labels = []
        parts = []
        colors = []
        cats = []
        path_lab = self.opt.dataroot
        for i, cat in enumerate(self.opt.categorys):

            if cat not in self.opt.selected_cats:
                continue

            cat = str(cat)
            for item in sorted(os.listdir(os.path.join(path_lab, cat, mode))):
                if item.find("label") != -1:
                    labels.append(os.path.join(cat, mode, item))
                    cats.append(i)
                if item.find("part") != -1:
                    parts.append(os.path.join(cat, mode, item))
                if item.find("label") != -1:
                    colors.append(os.path.join(cat, mode, item))


        assert len(labels)  == len(parts), "different len of images and labels %s - %s" % (len(parts), len(labels))
        return labels, parts, colors, cats, (path_lab, path_lab, path_lab)

    def transforms(self, label, part, color):

        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            # random augmentation
            if random.random() < 0.5:
                # resize
                new_width, new_height = (self.opt.load_size, self.opt.load_size)
                label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
                part = TR.functional.resize(part, (new_width, new_height), Image.NEAREST)
                color = TR.functional.resize(color, (new_width, new_height), Image.CUBIC)

                # rotate
                if random.random() < 0.5:
                    angle = random.randint(-15, 16)
                    label = TR.functional.rotate(label, angle, Image.NEAREST)
                    part = TR.functional.rotate(part, angle, Image.NEAREST)
                    color = TR.functional.rotate(color, angle, Image.CUBIC)

                # crop
                crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
                crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
                label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
                part = part.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
                color = color.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))

            else:
                new_width, new_height = (self.opt.crop_size, self.opt.crop_size)
                label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
                part = TR.functional.resize(part, (new_width, new_height), Image.NEAREST)
                color = TR.functional.resize(color, (new_width, new_height), Image.CUBIC)

            # random flip
            if random.random() < 0.5:
                label = TR.functional.hflip(label)
                part = TR.functional.hflip(part)
                color = TR.functional.hflip(color)
        else:
            new_width, new_height = (self.opt.crop_size, self.opt.crop_size)
            label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
            part = TR.functional.resize(part, (new_width, new_height), Image.NEAREST)
            color = TR.functional.resize(color, (new_width, new_height), Image.CUBIC)

        # to tensor
        label = TR.functional.to_tensor(label)
        part = TR.functional.to_tensor(part)
        color = TR.functional.to_tensor(color)

        color = TR.functional.normalize(color, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return label, part, color