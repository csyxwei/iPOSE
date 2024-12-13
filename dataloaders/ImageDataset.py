import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
from os.path import basename

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 224
        opt.crop_size = 224
        opt.label_nc = 3
        opt.contain_dontcare_label = False
        opt.semantic_nc = 7 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        gt = int(basename(self.images[idx]).split('_')[-2])
        if gt > 1000:
            gt = gt // 1000
        gt = torch.tensor(gt)
        return {"image": image, "target": gt, "name": self.labels[idx]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        # mode = 'test'
        images = []
        path_img = os.path.join(self.opt.dataroot)
        labels = []
        path_lab = os.path.join(self.opt.dataroot)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder, mode)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("image") != -1:
                    images.append(os.path.join(city_folder, mode, item))
                if item.find("label") != -1:
                    labels.append(os.path.join(city_folder, mode, item))
        assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace("_image.png", "") == labels[i].replace("_label.png", ""), \
                '%s and %s are not matching' % (images[i], labels[i])

        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        # resize
        image = TR.functional.resize(image, (self.opt.crop_size, self.opt.crop_size), Image.BICUBIC)
        label = TR.functional.resize(label, (self.opt.crop_size, self.opt.crop_size), Image.NEAREST)

        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.48145466, 0.4578275, 0.40821073),
                                        (0.26862954, 0.26130258, 0.27577711))
        return image, label