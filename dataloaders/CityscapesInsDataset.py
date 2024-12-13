import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import cv2

class CityscapesInsDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        opt.categorys = [24, 25, 26, 27, 28, 31, 32, 33]

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.instances, self.paths = self.list_images()
        self.process_ins()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        instance = Image.open(os.path.join(self.paths[2], self.instances[idx]))
        image, label, instance, isflip = self.transforms(image, label, instance)
        label = label * 255
        crop_params = torch.tensor(self.crop_params[idx])
        ins_num = torch.tensor(self.instance_num[idx])
        if isflip:
            c, h, w = instance.shape
            tmp = w - crop_params[:, 4]
            crop_params[:, 4] = w - crop_params[:, 3]
            crop_params[:, 3] = tmp

        return {"image": image, "label": label, "instance":instance, "crop_params": crop_params, "ins_num":ins_num, "name": self.images[idx]}


    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        instances = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))
                if item.find("instanceIds") != -1:
                    instances.append(os.path.join(city_folder, item))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        assert len(images) == len(instances), "different len of images and instances %s - %s" % (len(images), len(instances))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
            assert images[i].replace("_leftImg8bit.png", "") == instances[i].replace("_gtFine_instanceIds.png", ""), \
                '%s and %s are not matching' % (images[i], instances[i])

        return images, labels, instances, (path_img, path_lab, path_lab)

    def transforms(self, image, label, instance):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        instance = TR.functional.resize(instance, (new_width, new_height), Image.NEAREST)

        # flip
        isflip = False
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
                instance = TR.functional.hflip(instance)
                isflip = True
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        instance = np.array(instance)[:, :, None]
        instance = torch.from_numpy(instance.transpose((2, 0, 1)))

        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label, instance, isflip

    def process_ins(self):
        crop_params = []
        self.instance_num = []
        for index in range(len(self.instances)):
            params = []
            ins_np = np.array(Image.open(os.path.join(self.paths[2], self.instances[index])))
            idxes = np.unique(ins_np)
            idxes = [idx for idx in idxes if idx >= 1000]
            for idx in idxes:

                s = idx // 1000
                if not s in self.opt.categorys:
                    continue

                tmp = np.where(ins_np == idx, 255, 0)
                tmp = np.array(tmp, dtype=np.uint8)

                contours = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                if len(contours) > 1:
                    cntr = np.vstack(contours)
                else:
                    cntr = contours[0]

                if len(cntr) < 2:
                    continue

                hs, he = np.min(cntr[:, :, 1]), np.max(cntr[:, :, 1])
                ws, we = np.min(cntr[:, :, 0]), np.max(cntr[:, :, 0])

                h, w = tmp.shape  # 256x512

                # padding
                if (he - hs) % 2 == 1 and (he + 1) <= h:
                    he = he + 1
                if (he - hs) % 2 == 1 and (hs - 1) >= 0:
                    hs = hs - 1
                if (we - ws) % 2 == 1 and (we + 1) <= w:
                    we = we + 1
                if (we - ws) % 2 == 1 and (ws - 1) >= 0:
                    ws = ws - 1

                if he - hs < 2 or we - ws < 2:
                    continue

                params.append([idx, hs, he, ws, we, 1, 1])
            crop_params.append(params)
            self.instance_num.append(len(params))

        max_len = max(self.instance_num)
        self.crop_params = []
        for params in crop_params:
            p_len = len(params)
            if p_len < max_len:
                padded_params = params + [[0,0,0,0,0,1,1] for _ in range(max_len - p_len)]
            else:
                padded_params = params
            self.crop_params.append(padded_params)
