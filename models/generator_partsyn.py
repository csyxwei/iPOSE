import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import random
import os
from os.path import join
from models.modules import ConvBlock, DeConvBlock, CrossAttentionLayer


chs = [32, 32, 64, 64, 96, 128, 128, 256]

class Encoder(nn.Module):
    def __init__(self, input_dim, layers=3, ms_feature=False, res_num=0):
        super(Encoder, self).__init__()

        self.layers = layers
        self.ms_feature = ms_feature

        self.enc0 = ConvBlock(input_dim, chs[0], 7, 1, 3, res=res_num)

        for i in range(1, self.layers+1):
            setattr(self, f'enc{i}', ConvBlock(chs[i], chs[i + 1], 4, 2, 1, res=res_num))

        if self.ms_feature:
            self.ms_enc0 = ConvBlock(chs[self.layers+1], chs[self.layers+1], 4, 2, 1, res=res_num)
            self.ms_enc1 = ConvBlock(chs[self.layers+1], chs[self.layers+1], 4, 2, 1, res=res_num)

    def forward(self, x):

        for i in range(self.layers+1):
            x = getattr(self, f'enc{i}')(x)
        out = [x]
        if self.ms_feature:
            x = self.ms_enc0(x)
            out.append(F.interpolate(x, out[0].shape[2:], mode='bilinear'))
            x = self.ms_enc1(x)
            out.append(F.interpolate(x, out[0].shape[2:], mode='bilinear'))
        out = torch.cat(out, dim=1)
        return out

class Decoder(nn.Module):
    def __init__(self, layers=3, ms_feature=False, res_num=0):
        super(Decoder, self).__init__()

        self.layers = layers
        dchs = chs[:self.layers + 2][::-1]

        if ms_feature:
            dchs[0] = dchs[0] * 3

        for i in range(1, self.layers + 1):
            setattr(self, f'dec{i}', DeConvBlock(dchs[i - 1], dchs[i], 4, 2, 1, res=res_num))

    def forward(self, x):
        for i in range(1, self.layers + 1):
            x = getattr(self, f'dec{i}')(x)
        return x


class PartGeneratorBase(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def load_support(self, path):
        img_np = np.array(Image.open(path)).astype('float32')
        img_th = torch.from_numpy(img_np[None, None, :, :])
        img_th = torch.nn.functional.interpolate(img_th, (64, 64), mode='nearest')
        return img_th

    def load_support_list(self, n_support):

        self.dataroot = self.opt.dataroot

        mapping_dict = {k:k for k in range(1, 50)}
        mapping_dict[6] = 28 # bus ADE
        mapping_dict[7] = 26 # car ADE
        mapping_dict[15] = 24 # human ADE
        mapping_dict[25] = 24 # rider
        mapping_dict[31] = 19 # train City
        mapping_dict[32] = 14 # motor City
        mapping_dict[33] = 2 # bicycle City
        mapping_dict[35] = 26 # ADE car
        mapping_dict[39] = 26
        mapping_dict[40] = 13

        support_dir = join(self.dataroot, 'support_shape', str(n_support))

        support_dict = {}
        for k in mapping_dict:
            img_list = []
            k_dir = join(support_dir, str(mapping_dict[k]))
            if not os.path.exists(k_dir):
                continue
            files = sorted(os.listdir(k_dir))
            files = [os.path.join(k_dir, file) for file in files if file.endswith('part.png')]
            for file in files:
                img_list.append(self.load_support(file))

            k_dir = join(self.dataroot, str(mapping_dict[k]), 'train')
            if os.path.exists(k_dir):
                files = os.listdir(k_dir)
                files = [os.path.join(k_dir, file) for file in files if file.endswith('part.png')]
                for file in files:
                    if len(img_list) > 100:
                        continue
                    img_th = self.load_support(file)
                    if len(torch.unique(img_th)) != len(torch.unique(img_list[0])):
                        continue
                    img_list.append(img_th)
            support_dict[k] = img_list
            print(k, len(img_list))
        return support_dict

    def load_supports_random(self, random_support, n_support, nc):
        index_dict = {}
        if not self.training or not random_support:
            random_type = 0
        else:
            random_type = random.sample([0, 0, 0, 0, 0, 1, 2, 3], 1)[0]

        for k in self.opt.categorys:
            select_idx = list(range(n_support))
            if not random_type == 0:
                select_idx += random.sample(list(range(1, len(self.support_dict[k]))), min(n_support * 2, len(self.support_dict[k])-1))
                select_idx = random.sample(select_idx, n_support)
            index_dict[k] = select_idx

        support_list = []
        for i in range(n_support):
            support_i = []
            for cat in self.opt.categorys:
                img_th = self.support_dict[cat][index_dict[cat][i]]
                img_th_long = img_th.long()
                bs, _, h, w = img_th_long.size()
                input_label = torch.FloatTensor(bs, nc, h, w).zero_()
                img_th = input_label.scatter_(1, img_th_long, 1.0)
                support_i.append(img_th)

            support_i = torch.cat(support_i, dim=0)
            support_list.append(support_i)

        return support_list

class PartGeneratorAttn(PartGeneratorBase):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        input_dim = 3
        if self.opt.use_coord:
            input_dim = input_dim + 2

        self.max_n_support = 5
        self.n_support = opt.n_support

        self.layers = 3
        self.num_att_layers = opt.n_att_layers
        self.ms_feature = True

        self.ffn = True
        res_num = 0
        self.shape_encoder = Encoder(input_dim, self.layers, self.ms_feature, res_num)
        self.mask_encoder = Encoder(1, self.layers, self.ms_feature, res_num)

        self.decoder = Decoder(self.layers, self.ms_feature, res_num)

        att_ch = chs[self.layers + 1] * 3 if self.ms_feature else chs[self.layers + 1]
        for i in range(self.num_att_layers):
            setattr(self, f'catt{i}', CrossAttentionLayer(att_ch, att_ch, 1, self.ffn))

        setattr(self, f'conv_img', nn.Conv2d(chs[1], 1, 5, padding=2))

        self.k_support_dict = {}

        for i in [self.n_support]:
            support_dict = self.load_support_list(i)
            self.k_support_dict[i] = support_dict

    def forward_once(self, query, supports, gt):

        # get encoder feature
        enc_seg = self.shape_encoder(query)
        enc_gt = self.shape_encoder(gt[:, :-1])
        mask_gt = self.mask_encoder(gt[:, -1:])

        dsupps = []
        # get n templates encoder feature
        for i in range(self.n_support):
            enc_supp = self.shape_encoder(supports[i][:, :-1])
            mask_supp = self.mask_encoder(supports[i][:, -1:])
            dsupps.append([enc_supp, mask_supp])

        dec_syn = enc_seg
        dec_rec = enc_seg
        dec_gt = [enc_gt, mask_gt]

        if self.n_support > 1:
            t = random.sample(list(range(self.n_support - 1)), 1)[0]
            recsupps = dsupps[:t] + [dec_gt] + dsupps[t + 1:]
        else:
            recsupps = [dec_gt]

        for i in range(self.num_att_layers):
            dec_rec = getattr(self, f'catt{i}')(dec_rec, recsupps, image_mask_split=True)
            dec_syn = getattr(self, f'catt{i}')(dec_syn, dsupps, image_mask_split=True)

        dec_syn = self.decoder(dec_syn)
        dec_rec = self.decoder(dec_rec)

        x_syn = getattr(self, f'conv_img')(dec_syn)
        x_rec = getattr(self, f'conv_img')(dec_rec)

        return x_syn, x_rec

    def forward(self, input, cat, gt):
        random_supp = True

        self.support_dict = self.k_support_dict[self.n_support]
        self.supports = self.load_supports_random(random_supp, self.n_support, gt.shape[1])

        supports = []
        supports_shape = []
        for i in range(self.n_support):
            support_i = self.supports[i].to(input.device)[cat]
            support_i_shape = torch.argmax(support_i, dim=1, keepdim=True)
            support_i_shape = torch.where(support_i_shape > 0, torch.ones_like(support_i_shape), support_i_shape).detach()
            supports.append(support_i)
            supports_shape.append(support_i_shape)

        x_range = torch.linspace(-1, 1, input.shape[-1], device=input.device)
        y_range = torch.linspace(-1, 1, input.shape[-2], device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input.shape[0] * gt.shape[1], 1, -1, -1])
        x = x.expand([input.shape[0] * gt.shape[1], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        gt_shape = torch.argmax(gt, dim=1, keepdim=True)
        gt_shape = torch.where(gt_shape > 0, torch.ones_like(gt_shape), gt_shape).detach()

        gt_dim = gt.shape[1]
        aug = True
        if aug and self.training:
            idx = torch.randperm(gt_dim)
            gt = gt[:, idx].view(gt.size())
            supports = [support[:, idx].view(gt.size()) for support in supports]

        bs, c, h, w = input.shape
        seg_in = input.unsqueeze(1).repeat((1, gt_dim, 1, 1, 1)).view(bs * gt_dim, c, h, w)

        gt_shape = gt_shape.unsqueeze(1).repeat((1, gt_dim, 1, 1, 1)).view(bs * gt_dim, 1, h, w)
        gt_shape = gt_shape.repeat((1, c, 1, 1))
        gt_in = gt.view(bs * gt_dim, 1, h, w)
        if self.opt.use_coord:
            seg_in = torch.cat((coord_feat, seg_in), 1)
            gt_in = torch.cat((coord_feat, gt_shape, gt_in), 1)
        else:
            gt_in = torch.cat((gt_shape, gt_in), 1)

        supports_in = []
        for i in range(self.n_support):
            support_i_shape = supports_shape[i].unsqueeze(1).repeat((1, gt_dim, 1, 1, 1)).view(bs * gt_dim, 1, h, w)
            support_i_shape = support_i_shape.repeat((1, c, 1, 1))
            support_i_in = supports[i].view(bs * gt_dim, 1, h, w)
            if self.opt.use_coord:
                supports_in.append(torch.cat((coord_feat, support_i_shape, support_i_in), 1))
            else:
                supports_in.append(torch.cat((support_i_shape, support_i_in), 1))
        x_syn, x_rec = self.forward_once(seg_in, supports_in, gt_in)

        x_syn = x_syn.view(bs, gt_dim, h, w)
        x_rec = x_rec.view(bs, gt_dim, h, w)

        if self.training:
            return x_syn, x_rec, supports, gt
        else:
            return x_syn, supports, None, None, x_rec