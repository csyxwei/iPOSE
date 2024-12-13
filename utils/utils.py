import torch
import numpy as np
import random
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter  = (start_iter + 1) %  dataset_size
    return start_epoch, start_iter


def multi_acc(opt, model, models, dataloader, imsaver=None):
    t = 0
    all = 0

    t_map = {}
    all_map = {}
    count_map = {}

    model = model.eval()
    with torch.no_grad():
        for i, data_i in enumerate(dataloader):
            image, labels = models.preprocess_input(opt, data_i)

            label, shape_context, part, cat, w = labels
            y_pred = model(image, labels, 'generate', None)
            y_pred_softmax = torch.log_softmax(y_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            correct_pred = (y_pred_tags == torch.argmax(part, dim=1)).float()
            t += correct_pred.sum()
            all += len(correct_pred.view(-1))

            for ii, g in enumerate(cat):
                idx = g.item()
                if idx not in all_map:
                    all_map[idx] = len(correct_pred[ii].view(-1))
                else:
                    all_map[idx] = all_map[idx] + len(correct_pred[ii].view(-1))

                if idx not in count_map:
                    count_map[idx] = 1
                else:
                    count_map[idx] = count_map[idx] + 1

                if count_map[idx] < 20 and imsaver is not None:
                    imsaver((y_pred[ii:ii+1], part[ii:ii+1]), label[ii:ii+1], data_i["name"][ii:ii+1])

                if idx not in t_map:
                    t_map[idx] = correct_pred[ii].sum()
                else:
                    t_map[idx] = t_map[idx] + correct_pred[ii].sum()

            # if i < 2 and imsaver is not None:

    for k in t_map:
        print(k, t_map[k].item() / all_map[k])

    model = model.train()
    acc = t / all
    acc = acc * 100
    return acc


def multi_acc_ctr(opt, model, models, dataloader, imsaver=None):
    t = 0
    all = 0

    t_map = {}
    all_map = {}
    count_map = {}

    model = model.eval()
    with torch.no_grad():
        for i, data_i in enumerate(dataloader):
            image, labels = models.preprocess_input(opt, data_i)

            label, temps, real_image, cat = labels
            part = real_image[:, :, 1]

            y_pred = model(image, labels, 'generate', None)
            y_pred_softmax = torch.log_softmax(y_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            correct_pred = (y_pred_tags == torch.argmax(part, dim=1)).float()
            t += correct_pred.sum()
            all += len(correct_pred.view(-1))

            for ii, g in enumerate(cat):
                idx = g.item()
                if idx not in all_map:
                    all_map[idx] = len(correct_pred[ii].view(-1))
                else:
                    all_map[idx] = all_map[idx] + len(correct_pred[ii].view(-1))

                if idx not in count_map:
                    count_map[idx] = 1
                else:
                    count_map[idx] = count_map[idx] + 1

                if count_map[idx] < 20 and imsaver is not None:
                    imsaver((y_pred[ii:ii+1], part[ii:ii+1]), label[ii:ii+1], data_i["name"][ii:ii+1])

                if idx not in t_map:
                    t_map[idx] = correct_pred[ii].sum()
                else:
                    t_map[idx] = t_map[idx] + correct_pred[ii].sum()

            # if i < 2 and imsaver is not None:

    for k in t_map:
        print(k, t_map[k].item() / all_map[k])

    model = model.train()
    acc = t / all
    acc = acc * 100

    print(7, t_map[7].item() / all_map[7])

    x = 0
    y = 0
    for k in t_map:
        if k == 7:
            continue
        x += t_map[k].item()
        y += all_map[k]
    print('all', x/y)

    return acc


def multi_acc_withvis(opt, model, models, dataloader, imsaver=None):
    def save_im(im, path, mode, name, suffix):
        if mode == 'label':
            im = tens_to_lab(im[0], num_cl)
        elif mode == 'pred':
            if im[0].shape[0] == 1:
                im = im[0, 0]
            else:
                im = torch.argmax(im[0], dim=0)
            im = im.detach().cpu().numpy()
        else:
            im = tens_to_im(im[0]) * 255
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(path, name[0].split("/")[-1][:-10] + f'_{suffix}.png'))

    t = 0
    all = 0

    t_map = {}
    all_map = {}
    count_map = {}

    path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "vis")
    os.makedirs(path, exist_ok=True)

    path_syn = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "label")
    os.makedirs(path_syn, exist_ok=True)

    path_partsyn = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "partsyn")
    os.makedirs(path_partsyn, exist_ok=True)

    path_gt = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "part")
    os.makedirs(path_gt, exist_ok=True)

    path_label = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "image")
    os.makedirs(path_label, exist_ok=True)

    path_channel = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "channel")
    os.makedirs(path_channel, exist_ok=True)

    path_rec = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, "rec")
    os.makedirs(path_rec, exist_ok=True)

    num_cl = 9 # max(opt.label_nc + 2, opt.part_nc)

    model = model.eval()
    with torch.no_grad():
        for i, data_i in enumerate(dataloader):
            image, labels = models.preprocess_input(opt, data_i)

            label, shape_context, part, cat, w = labels
            y_pred, temps, flows, atts, x_rec = model(image, labels, 'generate_withvis', None)
            y_pred_softmax = torch.log_softmax(y_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            correct_pred = (y_pred_tags == torch.argmax(part, dim=1)).float()
            t += correct_pred.sum()
            all += len(correct_pred.view(-1))

            for ii, g in enumerate(cat):
                idx = g.item()
                if idx not in all_map:
                    all_map[idx] = len(correct_pred[ii].view(-1))
                else:
                    all_map[idx] = all_map[idx] + len(correct_pred[ii].view(-1))

                if idx not in count_map:
                    count_map[idx] = 1
                else:
                    count_map[idx] = count_map[idx] + 1

                if count_map[idx] < 300 and imsaver is not None:
                    for tem_idx in range(len(temps)):
                        temp = temps[tem_idx]
                        save_im(temp[ii:ii + 1], path, 'label', data_i["name"][ii:ii + 1], f'temp{tem_idx}')

                    save_im(y_pred[ii:ii + 1], path_syn, 'label', data_i["name"][ii:ii + 1], 'syn')
                    save_im(y_pred[ii:ii + 1], path_partsyn, 'pred', data_i["name"][ii:ii + 1], 'syn')
                    save_im(part[ii:ii + 1], path_gt, 'label', data_i["name"][ii:ii + 1], 'part')
                    save_im(label[ii:ii + 1], path_label, 'image', data_i["name"][ii:ii + 1], 'label')

                if idx not in t_map:
                    t_map[idx] = correct_pred[ii].sum()
                else:
                    t_map[idx] = t_map[idx] + correct_pred[ii].sum()


    for k in t_map:
        print(k, t_map[k].item() / all_map[k])

    model = model.train()
    acc = t / all
    acc = acc * 100

    print(7, t_map[7].item() / all_map[7])

    x = 0
    y = 0
    for k in t_map:
        if k == 7:
            continue
        x += t_map[k].item()
        y += all_map[k]
    print('all', x / y)

    return acc

def multi_acc_tmp(opt, models, model, dataloader, imsaver=None):
    t = 0
    all = 0
    count = 0

    t_map = {}
    all_map = {}
    count_map = {}

    model = model.eval()
    with torch.no_grad():
        for i, data_i in enumerate(dataloader):
            image, label = models.preprocess_input(opt, data_i)
            part = label[2]
            y_pred, y_predc, weight, part_batch, fake_batch, cat = model(image, label, 'generate_', None)

            y_pred_softmax = torch.log_softmax(fake_batch, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            correct_pred = (y_pred_tags == torch.argmax(part_batch, dim=1)).float()

            if len(cat) > 0:
                t += correct_pred.sum()
                all += len(correct_pred.view(-1))

            for ii, g in enumerate(cat):
                idx = g.item()
                if idx not in all_map:
                    all_map[idx] = len(correct_pred[ii].view(-1))
                else:
                    all_map[idx] = all_map[idx] + len(correct_pred[ii].view(-1))

                if idx not in t_map:
                    t_map[idx] = correct_pred[ii].sum()
                else:
                    t_map[idx] = t_map[idx] + correct_pred[ii].sum()

            for ii in range(len(y_pred)):
                if count < 40 and imsaver is not None:
                    imsaver((y_pred[ii:ii + 1], y_predc[ii:ii+1]), image[ii:ii + 1], data_i["name"][ii:ii + 1])
                    count = count + 1

    for k in t_map:
        print(k, t_map[k].item() / all_map[k])

    model = model.train()
    acc = t / all
    acc = acc * 100
    return acc

class results_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_part = os.path.join(path, "part")
        self.path_part_ori = os.path.join(path, "part_ori")
        self.path_to_save = {"label": self.path_label, "image": self.path_image, 'part': self.path_part, 'part_ori':self.path_part_ori}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        os.makedirs(self.path_part, exist_ok=True)
        os.makedirs(self.path_part_ori, exist_ok=True)
        self.num_cl = max(opt.label_nc + 2, opt.part_nc)

    def __call__(self, label, generated, name):

        if type(label) is tuple:
            assert len(label[0]) == len(generated)

        else:
            assert len(label) == len(generated)

        for i in range(len(generated)):
            if type(label) is tuple:
                im = tens_to_lab(label[0][i], self.num_cl)
                self.save_im(im, "label", name[i])
                im = tens_to_lab(label[1][i], 9)
                self.save_im(im, "part", name[i])
                part = label[1]
                part = torch.argmax(part, dim=1)
                part = part[i].detach().cpu().numpy()
                self.save_im(part, "part_ori", name[i])
            else:
                im = tens_to_lab(label[i], self.num_cl)
                self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt, name_list=None):
        self.name_list = ["Generator", "Vgg", 'G_ce', "D_fake", "D_real", "LabelMix", 'D_ce']
        if name_list is not None:
            self.name_list = name_list
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                if os.path.exists(self.path+"/losses.npy"):
                    self.losses[name] = np.load(self.path+"/losses.npy", allow_pickle = True).item()[name]
                else:
                    self.losses[name] = list()
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()

        if epoch % 50 == 0:
            print('Iter {}:'.format(epoch), end=' ')
            for name, loss in zip(self.name_list, losses):
                if loss is None:
                    continue
                print('{}: {} '.format(name, loss.detach().cpu().numpy()), end=', ')
            print()

        if epoch % self.freq_smooth_loss == self.freq_smooth_loss-1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i]/self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss-1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig,ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve])))*self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)),  dpi=600)
            plt.close(fig)

        fig,ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(model, model_, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with torch.no_grad():
        # for key in model.module.netEMA.state_dict():
        #     model.module.netEMA.state_dict()[key].data.copy_(
        #         model.module.netEMA.state_dict()[key].data * opt.EMA_decay +
        #         model.module.netG.state_dict()[key].data   * (1 - opt.EMA_decay)
        #     )
        ema_state = model.module.netEMA.state_dict()
        model_state = model.module.netG.state_dict()
        new_state = {}
        for key in ema_state:
            new_state[key] = ema_state[key] * opt.EMA_decay + model_state[key] * (1 - opt.EMA_decay)
        model.module.netEMA.load_state_dict(new_state, strict=False)
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with torch.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label = model_.preprocess_input(opt, data_i)
                fake = model.module.forward(None, label, 'generate', None)
                # fake = model(None, label, 'generate', None)
                num_upd += 1
                if num_upd > 50:
                    break


def save_networks(opt, cur_iter, model, latest=False, best=False, model_list=['netG', 'netD', 'netEMA', 'netE']):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        model_name = 'latest'
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        model_name = 'best'
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        model_name = cur_iter

    for name in model_list:
        sub_names = name.split('.')
        module = model.module
        for sub_name in sub_names:
            if hasattr(module, sub_name):
                module = getattr(module, sub_name)
            else:
                module = None
                break
        if module is not None:
            torch.save(module.state_dict(), path + '/%s_%s.pth' % (model_name, name.replace('net', '')))

class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images")+"/"
        self.opt = opt
        self.num_cl = max(opt.label_nc + 2, opt.part_nc)
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter, part=None):
        if type(label) is tuple:
            self.save_images(label[0], "label", cur_iter, is_label=True)
        else:
            self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        with torch.no_grad():
            model.eval()
            fake = model(None, label, 'generate', None)
            self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                if type(label) is tuple:
                    fake = model.module.netEMA(label[0])
                else:
                    fake = model.module.netEMA(label)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def visualize_batch_spd(self, model, image, label, cur_iter, part=None):
        if type(label) is tuple:
            self.save_images(label[0], "label", cur_iter, is_label=True)
        else:
            self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        with torch.no_grad():
            model.eval()
            fake = model(None, label, 'generate', None)
            self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                fake = model.module.netEMA(label)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def visualize_batch_ins(self, model, image, label, cur_iter, part=None):
        self.save_images(label[0], "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        with torch.no_grad():
            model.eval()
            fake, part = model(None, label, 'generate_withpart', None)
            self.save_images(fake, "fake", cur_iter)
            self.save_images(part, "part", cur_iter, is_label=True)
            model.train()


    def visualize_batch_part_syn(self, model, image, label, cur_iter):
        self.save_images(label[0], "label", cur_iter)
        # self.save_images(label[2], "real", cur_iter, is_label=True)
        with torch.no_grad():
            model.eval()
            fake = model(image, label, 'generate_withvis', None)
            self.save_images(fake[0], "fake", cur_iter, is_label=True)
            self.save_images(fake[-1], "rec", cur_iter, is_label=True)
            self.save_images(fake[1][0], "temp0", cur_iter, is_label=True)
            if len(fake[1]) > 1:
                self.save_images(fake[1][1], "temp1", cur_iter, is_label=True)
            # self.save_images(fake[-1], "real", cur_iter, is_label=True)
        model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i+1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path+str(cur_iter)+"_"+name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    if tens.shape[0] == 1:
        pass
    else:
        tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 9:
        #  https://matplotlib.org/stable/tutorials/colors/colormaps.html
        GnBu_Cmap = plt.cm.GnBu
        cmap = []
        for i in range(7):
            map = (i - 0) / 6
            cmap.append([GnBu_Cmap(map)[0] * 255, GnBu_Cmap(map)[1] * 255, GnBu_Cmap(map)[2] * 255])
        # cmap[0] = [242, 242, 242]
        cmap = np.array(cmap, dtype=np.uint8)
    elif N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap




