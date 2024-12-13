import random
import numpy as np
from models.sync_batchnorm import DataParallelWithCallback
import models.generator_ipose as generators
import models.discriminator as discriminators
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses
import models.clip_loss as clip_loss


class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.OASIS_Generator(opt)
        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()

        state_path = os.path.join(opt.checkpoints_dir, 'partsyn_model', 'models', 'best_G.pth')

        state_dict = torch.load(state_path)
        self.netG.partnet.model.load_state_dict(state_dict)
        print('load from', state_path)
        if not opt.ft:
            for name, p in self.netG.partnet.model.named_parameters():
                p.requires_grad = False

        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()

        #--- perceptual loss ---#
        if opt.phase == "train":
            self.criterionL1 = nn.L1Loss()
            if opt.add_vgg_loss:
                self.VGG_loss = clip_loss.CLIPLoss(opt)
            if opt.use_globalD:
                self.FloatTensor = torch.cuda.FloatTensor if opt.gpu_ids != "-1" \
                    else torch.FloatTensor
                self.ByteTensor = torch.cuda.ByteTensor if opt.gpu_ids != "-1" \
                    else torch.ByteTensor
                self.criterionGAN = losses.GANLoss('hinge', tensor=self.FloatTensor, opt=self.opt)

    def forward(self, image, label_in, mode, losses_computer, z=None, cur_iter=None, z_part=None):
        # Branching is applied to be compatible with DataParallel
        label = label_in[0]

        if mode == "losses_G":
            loss_G = 0

            fake = self.netG(label_in)
            output_D = self.netD(fake)

            if self.opt.use_globalD:
                loss_G_adv_g = self.criterionGAN(output_D[1], True, for_discriminator=False) * 1
                loss_G_adv_d = losses_computer.loss(output_D[0], label, for_real=True)
                loss_G_adv = loss_G_adv_g + loss_G_adv_d
            else:
                loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv

            zzz = random.sample([0, 1], 1)[0]

            if self.opt.add_vgg_loss and zzz == 1:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            elif self.opt.add_vgg_loss and zzz == 0:
            # if self.opt.add_vgg_loss:
                #######
                merged_params = self.get_cropparams(label_in[1], label_in[2], label_in[3])
                if len(merged_params) == 0:
                    loss_G_vgg = torch.zeros_like(loss_G_adv).detach()
                else:
                    select_num = min(10, len(merged_params))
                    select_index = random.sample(list(range(len(merged_params))), select_num)
                    select_params = np.array((merged_params))[select_index]
                    cropped_images = self.decompose(image, select_params)
                    cropped_fakes = self.decompose(fake, select_params)
                    cropped_images = torch.cat(cropped_images, 0)
                    cropped_fakes = torch.cat(cropped_fakes, 0)

                    grad_lambda = False
                    if grad_lambda and cur_iter is not None and cur_iter < 2000:
                        lambda_vgg = 2
                    elif grad_lambda and cur_iter is not None and cur_iter < 4000:
                        lambda_vgg = 2 + (cur_iter - 2000) / 2000 * (self.opt.lambda_vgg - 2)
                    else:
                        lambda_vgg = self.opt.lambda_vgg

                    # cropped_images = image
                    # cropped_fakes = fake
                    # lambda_vgg = self.opt.lambda_vgg

                    loss_G_vgg = lambda_vgg * self.VGG_loss(cropped_fakes, cropped_images)
                    loss_G += loss_G_vgg
                # loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                # loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg, None]

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(label_in)
            output_D_fake = self.netD(fake)
            if self.opt.use_globalD:
                loss_D_fake_g = self.criterionGAN(output_D_fake[1], False, for_discriminator=True) * 1
                loss_D_fake_d = losses_computer.loss(output_D_fake[0], label, for_real=False)
                loss_D_fake = loss_D_fake_g + loss_D_fake_d
            else:
                loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            if self.opt.use_globalD:
                loss_D_real_g = self.criterionGAN(output_D_real[1], True, for_discriminator=True) * 1
                loss_D_real_d = losses_computer.loss(output_D_real[0], label, for_real=True)
                loss_D_real = loss_D_real_g + loss_D_real_d
            else:
                loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                if self.opt.use_globalD:
                    output_D_mixed = output_D_mixed[0]
                    output_D_fake = output_D_fake[0]
                    output_D_real = output_D_real[0]
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                                     output_D_fake, output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm, None]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label_in)
                else:
                    fake = self.netEMA(label_in)
            return fake

        if mode == "generate_withpart":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake, part = self.netG(label_in, return_part=True)
                else:
                    fake, part = self.netEMA(label_in, return_part=True)
            return fake, part

        if mode == "generate_withpartz":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake, part = self.netG(label_in, z=z, return_part=True, z_part=z_part)
                else:
                    fake, part = self.netEMA(label_in, z=z, return_part=True, z_part=z_part)
            return fake, part

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

    def get_cropparams(self, instance, crop_params, ins_num):
        params = []
        ins_np = instance.cpu().numpy()
        for b in range(len(ins_np)):
            for ii in range(ins_num[b]):
                idx, hs, he, ws, we, H, W = crop_params[b][ii]
                idx, hs, he, ws, we, H, W = int(idx), int(hs), int(he), int(ws), int(we), int(H), int(W)
                params.append([b, idx, hs, he, ws, we, H, W])
        return params

    def decompose(self, image, merged_params):
        cropped_images = []
        for param in merged_params:
            b, idx, hs, he, ws, we, H, W = param
            ins_th = image[b:b+1, :, :, :]
            cropped_image = ins_th[:, :, hs:he, ws:we].float()
            _, _, h, w = cropped_image.shape
            if H > W:
                r = H / W
                new_h = int(h * r)
                new_w = w
                cropped_image = torch.nn.functional.interpolate(cropped_image, (new_h, new_w), mode='bilinear')
            elif W > H:
                r = W / H
                new_h = h
                new_w = int(w * r)
                cropped_image = torch.nn.functional.interpolate(cropped_image, (new_h, new_w), mode='bilinear')
            else:
                cropped_image = cropped_image

            hh, ww = cropped_image.shape[2:]

            # hh, ww = he - hs, we - ws
            if hh > ww:
                r = (hh - ww) // 2
                cropped_image = torch.nn.functional.pad(cropped_image, (r, r, 0, 0, 0, 0, 0, 0))
            elif ww > hh:
                r = (ww - hh) // 2
                cropped_image = torch.nn.functional.pad(cropped_image, (0, 0, r, r, 0, 0, 0, 0))
            else:
                cropped_image = cropped_image

            cropped_image = torch.nn.functional.interpolate(cropped_image, (128, 128), mode='nearest')
            cropped_images.append(cropped_image)


        return cropped_images


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
        data['instance'] = data['instance'].cuda()
        data['crop_params'] = data['crop_params'].cuda()
        data['ins_num'] = data['ins_num'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'], (input_semantics, data['instance'], data['crop_params'], data['ins_num'])


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim = 1, keepdim = True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0,2,(1,)).to(label.device)
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map