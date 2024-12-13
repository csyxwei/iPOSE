import torch
import numpy as np
import torch.nn as nn
import models.norms as norms
import torch.nn.functional as F
from models.generator_partsyn2 import PartGeneratorAttn

class PartGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = PartGeneratorAttn(opt)
        self.part_dim = 9
        self.fea_dim = opt.part_nc * self.part_dim

        self.load_size = 64
        self.catgory_dict = {}
        for i, v in enumerate(self.opt.categorys):
            self.catgory_dict[v] = i

    def forward(self, input, instance, crop_params, ins_num):

        composed_fea = torch.zeros((instance.shape[0], self.fea_dim, instance.shape[2], instance.shape[3]), device=instance.device)
        composed_ins = torch.zeros((instance.shape[0], 1, instance.shape[2], instance.shape[3]), device=instance.device)
        composed_part = torch.zeros((instance.shape[0], self.part_dim, instance.shape[2], instance.shape[3]), device=instance.device)

        merged_params = self.get_cropparams(instance, crop_params, ins_num)

        if self.training and self.opt.ft:
        # if self.training:
            import random
            select_num = min(instance.shape[0] * 2, len(merged_params))
            select_index = random.sample(list(range(len(merged_params))), select_num)
            merged_params = np.array((merged_params))[select_index]

        if len(merged_params) == 0:
            pass
        else:
            import math
            batch_size = 10
            iters = math.ceil(len(merged_params) / batch_size)
            for i in range(iters):
                params, cropped_instances, batches, semantics = self.decompose(instance, merged_params[i*batch_size:(i+1)*batch_size])

                batch_cropped_instances = torch.cat(cropped_instances, dim=0).detach().float()
                batch_semantics = torch.tensor(semantics, device=instance.device)
                batch_cropped_instances3 = torch.cat([batch_cropped_instances] * 3, dim=1)
                x_syn, x_fea = self.model.forward(batch_cropped_instances3, batch_semantics, return_fea=True)
                composed_fea, composed_ins, composed_part = self.compose(composed_fea, composed_ins, composed_part, x_fea, batch_cropped_instances, x_syn, batches, params)

        return composed_fea, composed_part

    def get_cropparams(self, instance, crop_params, ins_num):
        params = []
        ins_np = instance.cpu().numpy()
        for b in range(len(ins_np)):
            for ii in range(ins_num[b]):
                idx, hs, he, ws, we, H, W = crop_params[b][ii]
                idx, hs, he, ws, we, H, W = int(idx), int(hs), int(he), int(ws), int(we), int(H), int(W)
                params.append([b, idx, hs, he, ws, we, H, W])
        return params

    def decompose(self, instance, merged_params):
        cropped_instances = []
        batches = []
        semantics = []
        params = []
        for param in merged_params:
            b, idx, hs, he, ws, we, H, W = param
            ins_th = instance[b:b+1, 0:1, :, :]

            hs = max(0, hs - 2)
            ws = max(ws - 2, 0)
            he = min(he + 2, ins_th.shape[2])
            we = min(we + 2, ins_th.shape[3])

            if (he - hs) % 2 == 1 and (he + 1) <= ins_th.shape[2]:
                he = he + 1
            if (he - hs) % 2 == 1 and (hs - 1) >= 0:
                hs = hs - 1
            if (we - ws) % 2 == 1 and (we + 1) <= ins_th.shape[3]:
                we = we + 1
            if (we - ws) % 2 == 1 and (ws - 1) >= 0:
                ws = ws - 1

            if 'ade20k' in self.opt.dataset_mode or 'coco' in self.opt.dataset_mode:
                new_idx = idx % 1000
            else:
                new_idx = idx

            ins_th = torch.where(ins_th == float(new_idx), torch.ones_like(ins_th), torch.zeros_like(ins_th))
            cropped_instance = ins_th[:, :, hs:he, ws:we].float()
            location_param = [hs, he, ws, we, H, W]

            _, _, h, w = cropped_instance.shape

            if H > W:
                r = H / W
                new_h = int(h * r)
                new_w = w
                cropped_instance = F.interpolate(cropped_instance, (new_h, new_w), mode='nearest')
            elif W > H:
                r = W / H
                new_h = h
                new_w = int(w * r)
                cropped_instance = F.interpolate(cropped_instance, (new_h, new_w), mode='nearest')
            else:
                cropped_instance = cropped_instance

            hh, ww = cropped_instance.shape[2:]

            # hh, ww = he - hs, we - ws
            if hh > ww:
                r = (hh - ww) // 2
                cropped_instance = F.pad(cropped_instance, (r, r, 0, 0, 0, 0, 0, 0))
                crop_param = [0, 0, r, -r]
                resize_param = hh
            elif ww > hh:
                r = (ww - hh) // 2
                cropped_instance = F.pad(cropped_instance, (0, 0, r, r, 0, 0, 0, 0))
                crop_param = [r, -r, 0, 0]
                resize_param = ww
            else:
                crop_param = [0, 0, 0, 0]
                resize_param = hh

            cropped_instance = F.interpolate(cropped_instance, (self.load_size, self.load_size), mode='nearest')

            s = idx // 1000
            semantics.append(self.catgory_dict[s])
            cropped_instances.append(cropped_instance)

            batches.append(b)
            params.append([location_param, crop_param, resize_param])

        return params, cropped_instances, batches, semantics

    def compose(self, composed_fea, composed_ins, composed_part, features, instances, parts, batches, params):

        for idx in range(len(batches)):
            b = batches[idx]
            location_param, crop_param, resize_param = params[idx]
            fea = features[idx:idx+1]
            instance = instances[idx:idx+1]
            part = parts[idx:idx+1]
            new_h, new_w = resize_param, resize_param
            resized_fea = F.interpolate(fea, (new_h, new_w), mode='bilinear')
            resized_ins = F.interpolate(instance, (new_h, new_w), mode='nearest')
            resized_part = F.interpolate(part, (new_h, new_w), mode='nearest')

            cropped_fea = resized_fea[:, :, crop_param[0]:new_h + crop_param[1], crop_param[2]:new_w + crop_param[3]]
            cropped_ins = resized_ins[:, :, crop_param[0]:new_h + crop_param[1], crop_param[2]:new_w + crop_param[3]]
            cropped_part = resized_part[:, :, crop_param[0]:new_h + crop_param[1], crop_param[2]:new_w + crop_param[3]]

            hs, he, ws, we, H, W = location_param

            if H != W:
                cropped_fea = F.interpolate(cropped_fea, (he - hs, we - ws), mode='bilinear')
                cropped_ins = F.interpolate(cropped_ins, (he - hs, we - ws), mode='nearest')
                cropped_part = F.interpolate(cropped_part, (he - hs, we - ws), mode='nearest')

            composed_fea[b:b + 1, :, hs:he, ws:we] = composed_fea[b:b+1, :, hs:he, ws:we] * (1 - cropped_ins) + cropped_fea * cropped_ins
            composed_ins[b:b + 1, :, hs:he, ws:we] = composed_ins[b:b + 1, :, hs:he, ws:we] * (1 - cropped_ins) + cropped_ins * cropped_ins
            composed_part[b:b + 1, :, hs:he, ws:we] = composed_part[b:b + 1, :, hs:he, ws:we] * (1 - cropped_ins) + cropped_part * cropped_ins

        return composed_fea, composed_ins, composed_part

class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])


        self.conv_part = nn.Sequential(nn.Conv2d(opt.part_nc * 9, 32, kernel_size=3, stride=1, padding=1),
                                               nn.InstanceNorm2d(32),
                                               nn.LeakyReLU(0.2))
        self.conv_nos = nn.ModuleList([])
        self.mlp = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                               nn.InstanceNorm2d(32),
                                               nn.LeakyReLU(0.2))

        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            if i < 3:
                inc = 33
            else:
                inc = 32
            self.conv_nos.append(nn.Sequential(nn.ConvTranspose2d(inc, 32, kernel_size=4, stride=2, padding=1),
                                               nn.InstanceNorm2d(32),
                                               nn.LeakyReLU(0.2)))
        input_dim = self.opt.semantic_nc
        if not self.opt.no_3dnoise:
            input_dim = input_dim + self.opt.z_dim

        self.fc = nn.Conv2d(input_dim, 16 * ch, 3, padding=1)
        self.partnet = PartGenerator(opt)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None, return_part=False, z_part=None, ):

        if not self.opt.ft:
            with torch.no_grad():
                part_fea, part_syn = self.partnet(input[0], input[1], input[2], input[3])
                mask = torch.sum(torch.argmax(part_syn, dim=1, keepdim=True), dim=1, keepdim=True) > 0
                mask = mask.float().detach()
        else:
            part_fea, part_syn = self.partnet(input[0], input[1], input[2], input[3])
            mask = torch.sum(torch.argmax(part_syn, dim=1, keepdim=True), dim=1, keepdim=True) > 0
            mask = mask.float().detach()

        semmap = torch.argmax(input[0], dim=1, keepdim=True).float()
        
        if not self.opt.ft:
            part_fea = part_fea.detach()

        seg = input[0]

        part_fea = self.conv_part(part_fea)

        if self.opt.gpu_ids != "-1":
            seg.cuda()

        dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"

        if z_part is not None:
            part_nos = z_part.to(dev)
        else:
            part_nos = torch.randn(seg.size(0), 31, self.init_W, self.init_H, dtype=torch.float32, device=dev) * 1

        ss = F.interpolate(semmap, (part_nos.size(2), part_nos.size(3)), mode='nearest')
        part_nos = torch.cat((part_nos, ss), dim=1)
        part_nos = self.mlp(part_nos)

        if not self.opt.no_3dnoise:
            if z is None:
                dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
                z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev) * 1
                # z = self.mapping(z)
                z = z.view(z.size(0), self.opt.z_dim, 1, 1)
                z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            else:
                dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
                z = z.to(dev)
                if len(z.shape) == 2:
                    z = z.view(z.size(0), self.opt.z_dim, 1, 1)
                    z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))

            seg = torch.cat((z, seg), dim=1)

        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):

            x = self.body[i](x, seg, part_fea, part_nos, mask)

            if i < self.opt.num_res_blocks - 1:
                x = self.up(x)
                if i < 3:
                    ss = F.interpolate(semmap, (part_nos.size(2), part_nos.size(3)), mode='nearest')
                    part_nos = torch.cat((part_nos, ss), dim=1)
                part_nos = self.conv_nos[i](part_nos)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        if return_part:
            return x, part_syn
        return x

class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim
        if opt.use_edge:
            spade_conditional_input_dims += 1

        norm_func = norms.SPADE

        self.conv_p = sp_norm(nn.Conv2d(fin, fin, kernel_size=3, padding=1))
        self.norm_p = norms.SPADE3(opt, fin, 32)

        self.norm_0 = norm_func(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norm_func(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norm_func(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, part, part_nos, mask=None):
        x = self.conv_p(self.activ(self.norm_p(x, part, part_nos, mask)))
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out