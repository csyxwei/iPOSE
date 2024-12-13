import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import cv2


class Ade20kInsDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        if opt.phase == "test" or for_metrics:
            opt.load_size = 256
        else:
            opt.load_size = 286


        opt.crop_size = 256
        opt.label_nc = 150
        opt.contain_dontcare_label = True
        opt.semantic_nc = 151 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        opt.categorys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 36, 38, 39]

        opt.select_categorys = [24, 26, 36, 11, 9, 18, 4, 28, 27, 20, 1, 5, 14, 2, 38, 39]

        self.city2ade = {26: 9,
                         24: 4,
                         36: 68,
                         11: 6,
                         9: 8,
                         18: 11,
                         4: 48,
                         28: 50,
                         27: 53,
                         20: 58,
                         1: 59,
                         5: 64,
                         14: 74,
                         2: 83,
                         38: 80,
                         39: 65,}

        self.load_size = 256 # opt.load_size
        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.instances, self.paths = self.list_images()

        self.ispreload = False

        if self.ispreload:
            self.preload()

        if os.path.exists(os.path.join(opt.dataroot, 'training_dict.npy')):
            self.preprocess_ins()
        else:
            self.process_ins()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        if self.ispreload:
            image = Image.fromarray(self.preload_images[idx])
            label = Image.fromarray(self.preload_labels[idx])
            instance = Image.fromarray(self.preload_instances[idx])
        else:
            image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
            label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
            instance = Image.open(os.path.join(self.paths[2], self.instances[idx]))

        image, label, instance, isflip, loc = self.transforms(image, label, instance)

        crop_params = torch.tensor(self.crop_params[idx])

        c, h, w = instance.shape

        crop_params[:, 1] = torch.clamp(crop_params[:, 1] - loc[1], 0, h)
        crop_params[:, 2] = torch.clamp(crop_params[:, 2] - loc[1], 0, h)

        crop_params[:, 3] = torch.clamp(crop_params[:, 3] - loc[0], 0, w)
        crop_params[:, 4] = torch.clamp(crop_params[:, 4] - loc[0], 0, w)

        new_crop_params = []
        for i in range(self.instance_num[idx]):
            param = crop_params[i:i+1]

            if param[0, 2] <= 0 or param[0, 4] <= 0:
                continue
            if param[0, 1] >= h or param[0, 3] >= w:
                continue
            hs, he, ws, we = param[0, 1], param[0, 2], param[0, 3], param[0, 4]

            # 偶数边长
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

            param[0, 1], param[0, 2], param[0, 3], param[0, 4] = hs, he, ws, we

            new_crop_params.append(param)

        if len(new_crop_params) > 0:
            new_crop_params = torch.cat(new_crop_params, 0)
            ins_num = torch.tensor(len(new_crop_params))
            new_crop_params = torch.cat((new_crop_params, crop_params[len(new_crop_params):]), 0)
        else:
            new_crop_params = crop_params
            ins_num = torch.tensor(0)

        if isflip:
            tmp = w - new_crop_params[:, 4]
            new_crop_params[:, 4] = w - new_crop_params[:, 3]
            new_crop_params[:, 3] = tmp

        label = label * 255
        instance = instance[1:2] * 255

        return {"image": image, "label": label, "instance": instance, "crop_params": new_crop_params, "ins_num": ins_num,
                "name": self.images[idx]}

    def list_images(self):
        mode = "validation" if self.opt.phase == "test" or self.for_metrics else "training"
        path_img = os.path.join(self.opt.dataroot, "images", mode)
        path_lab = os.path.join(self.opt.dataroot, "annotations", mode)
        path_ins = os.path.join(self.opt.dataroot, "annotations_instance", mode)
        img_list = os.listdir(path_img)
        lab_list = os.listdir(path_lab)
        ins_list = os.listdir(path_ins)

        img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
        lab_list = [filename for filename in lab_list if ".png" in filename or ".jpg" in filename]
        ins_list = [filename for filename in ins_list if ".png" in filename or ".jpg" in filename]

        images = sorted(img_list)
        labels = sorted(lab_list)
        instances = sorted(ins_list)
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        assert len(images) == len(instances), "different len of images and instances %s - %s" % (len(images), len(instances))

        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
            assert os.path.splitext(images[i])[0] == os.path.splitext(instances[i])[0], '%s and %s are not matching' % (images[i], instances[i])

        if not 'sample' in self.opt.results_dir:
            return images, labels, instances, (path_img, path_lab, path_ins)

        return images, labels, instances, (path_img, path_lab, path_ins)

    def transforms(self, image, label, instance):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.load_size, self.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        instance = TR.functional.resize(instance, (new_width, new_height), Image.NEAREST)

        # crop
        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        instance = instance.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))

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
        instance = TR.functional.to_tensor(instance)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label, instance, isflip, (crop_x, crop_y)

    def process_ins(self):
        crop_params = []
        self.instance_num = []
        for index in range(len(self.instances)):
            params = []
            label = np.array(Image.open(os.path.join(self.paths[2], self.instances[index])))
            H, W = label.shape[:2]
            label = cv2.resize(label, (self.load_size, self.load_size), interpolation=cv2.INTER_NEAREST)
            for cat in self.opt.select_categorys:

                s_mask = label[:, :, 0]
                i_mask = label[:, :, 1]

                ins_np = np.where(s_mask == self.city2ade[cat], i_mask, -1)
                idxes = np.unique(ins_np)

                for idx in idxes:
                    if idx == -1:
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

                    h, w = tmp.shape

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

                    entropyed_idx = cat * 1000 + idx

                    params.append([entropyed_idx, hs, he, ws, we, H, W])

            crop_params.append(params)
            self.instance_num.append(len(params))

        max_len = max(self.instance_num)
        self.crop_params = []
        for params in crop_params:
            p_len = len(params)
            if p_len < max_len:
                padded_params = params + [[0, 0, 0, 0, 0, 0, 0] for _ in range(max_len - p_len)]
            else:
                padded_params = params
            self.crop_params.append(padded_params)

    def preprocess_ins(self):
        mode = "validation" if self.opt.phase == "test" or self.for_metrics else "training"
        param_dict = np.load(os.path.join(self.opt.dataroot, f'{mode}_dict.npy'), allow_pickle=True).item()
        crop_params = []
        self.instance_num = []
        for index in range(len(self.instances)):
            params = []
            pre_params = param_dict[os.path.basename(self.instances[index])]
            for cat in self.opt.select_categorys:
                for pre_param in pre_params:
                    if pre_param[0] == self.city2ade[cat]:
                        s_idx, i_idx, hs, he, ws, we, H, W = pre_param
                        entropyed_idx = cat * 1000 + i_idx
                        params.append([entropyed_idx, hs, he, ws, we, H, W])
            crop_params.append(params)
            self.instance_num.append(len(params))

        max_len = max(self.instance_num)
        self.crop_params = []
        for params in crop_params:
            p_len = len(params)
            if p_len < max_len:
                padded_params = params + [[0, 0, 0, 0, 0, 0, 0] for _ in range(max_len - p_len)]
            else:
                padded_params = params
            self.crop_params.append(padded_params)

    def preload(self):

        self.preload_images = []
        self.preload_labels = []
        self.preload_instances = []

        for img_path, label_path, instance_path in zip(self.images, self.labels, self.instances):
            image = Image.open(os.path.join(self.paths[0], img_path)).convert('RGB')
            label = Image.open(os.path.join(self.paths[1], label_path))
            instance = Image.open(os.path.join(self.paths[2], instance_path))

            image = np.array(image)
            label = np.array(label)
            instance = np.array(instance)

            self.preload_images.append(image.astype('uint8'))
            self.preload_labels.append(label.astype('uint8'))
            self.preload_instances.append(instance.astype('uint8'))
