from models.sync_batchnorm import DataParallelWithCallback
import models.generator_partsyn as generators
import models.discriminator as discriminators
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import torch.autograd as autograd
import models.losses as losses
import torch.nn.functional as F

class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.PartGeneratorAttn(opt)

        self.print_parameter_count()
        self.init_networks()
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

        self.criterionTV = losses.TVLoss()
        self.criterionPair = losses.PairwiseLoss()
        self.criterionCE = torch.nn.CrossEntropyLoss()
        self.criterionBCE = torch.nn.BCELoss()

    def forward(self, image, labels, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel

        input_semantics, real_image, cat, weight = labels

        weight = F.softmax(weight, 0) * input_semantics.shape[0]

        if mode == "losses_G":

            ###### For Attn Model
            fake_image, fake_rec, _, real_aug = self.netG(input_semantics, cat, real_image)
            loss_list = []
            b, c, h, w = fake_image.shape
            weight_bce = weight.view(b, 1, 1, 1).repeat(1, c, h, w)
            loss_bce = F.binary_cross_entropy(torch.sigmoid(fake_image), real_aug)
            loss_bce_rec = F.binary_cross_entropy(torch.sigmoid(fake_rec), real_aug)
            loss_G = loss_bce + loss_bce_rec
            loss_list = loss_list + [loss_bce, loss_bce_rec]

            weight_ce = weight.view(b, 1, 1).repeat(1, h, w)
            loss_ce = F.cross_entropy(fake_image, torch.argmax(real_aug, dim=1), reduction='none')
            loss_ce = torch.mean(loss_ce)
            loss_ce_rec = F.cross_entropy(fake_rec, torch.argmax(real_aug, dim=1), reduction='none')
            loss_ce_rec = torch.mean(loss_ce_rec)

            loss_G = loss_G + loss_ce + loss_ce_rec
            loss_list = loss_list + [loss_ce, loss_ce_rec]

            return loss_G, loss_list


        if mode == "generate":
            with torch.no_grad():
                fake = self.netG(input_semantics, cat, real_image)[0]
            return fake

        if mode == "generate_withvis":
            with torch.no_grad():
                fake = self.netG(input_semantics, cat, real_image)
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))

        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG]
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
            networks = [self.netG]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    # move to GPU and change data types
    data['part'] = data['part'].cuda()
    label_map = data['part'].long()
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    cat = data['cat'].long().cuda()
    w = data['w'].float().cuda()

    return data['color'], (data['label'], input_semantics, cat, w)


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim = 1, keepdim = True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0,2,(1,), device=label.device)
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map

def gradient_penalty(f, real, fake=None):
    def interpolate(a, b=None):
        if b is None:  # interpolation in DRAGAN
            beta = torch.rand_like(a)
            b = a + 0.5 * a.var().sqrt() * beta
        alpha = torch.rand(a.size(0), 1, 1, 1)
        alpha = alpha.to(real.device)
        inter = a + alpha * (b - a)
        return inter

    x = interpolate(real, fake).requires_grad_(True)
    pred = f(x)
    if isinstance(pred, tuple):
        pred = pred[0]
    grad = autograd.grad(
        outputs=pred, inputs=x,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.size(0), -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp