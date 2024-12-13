import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

dataroot = '/home/weiyuxiang/datasets/COCO'

mode = "val"  # train/val
load_size = 256

path_img = os.path.join(dataroot, mode+"_img")
path_lab = os.path.join(dataroot, mode+"_label")
path_ins = os.path.join(dataroot, mode + "_inst")

img_list = sorted(os.listdir(path_img))
lab_list = sorted(os.listdir(path_lab))
ins_list = sorted(os.listdir(path_ins))

img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
lab_list = [filename for filename in lab_list if ".png" in filename or ".jpg" in filename]
ins_list = [filename for filename in ins_list if ".png" in filename or ".jpg" in filename]


crop_params = {}

for index in tqdm(range(len(ins_list))):
    params = []

    label = np.array(Image.open(os.path.join(path_lab, lab_list[index])))
    instance = np.array(Image.open(os.path.join(path_ins, ins_list[index])))
    image = np.array(Image.open(os.path.join(path_img, img_list[index])).convert('RGB'))

    H, W = label.shape[:2]

    image = cv2.resize(image, (load_size, load_size), interpolation=cv2.INTER_CUBIC)

    label = cv2.resize(label, (load_size, load_size), interpolation=cv2.INTER_NEAREST)
    instance = cv2.resize(instance, (load_size, load_size), interpolation=cv2.INTER_NEAREST)

    s_mask = label
    i_mask = instance

    s_idxes = np.unique(s_mask)

    for s_idx in s_idxes:
        ins_np = np.where(s_mask == s_idx, i_mask, -1)
        ins_idxes = np.unique(ins_np)
        for ins_idx in ins_idxes:
            if ins_idx == -1:
                continue
            tmp = np.where(ins_np == ins_idx, 255, 0)
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

            params.append([s_idx, ins_idx, hs, he, ws, we, H, W])
    crop_params[os.path.basename(lab_list[index])] = params

np.save(os.path.join(dataroot, f'{mode}_dict.npy'), crop_params)

