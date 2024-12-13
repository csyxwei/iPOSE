import torch.nn.utils.spectral_norm as spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch

class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, return_param=False):
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        if return_param:
            return gamma, beta

        normalized = self.first_norm(x)
        out = normalized * (1 + gamma) + beta
        return out


class SPADE2(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 32
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, 1, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, 1, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        normalized = self.first_norm(x)
        out = normalized * (1 + gamma) + beta
        return out


class SPADE3(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 32
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, 1, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, 1, kernel_size=ks, padding=pw)

        self.mlp_shared_nos = nn.Sequential(
            nn.Conv2d(32, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma_nos = nn.Conv2d(nhidden, 1, kernel_size=ks, padding=pw)
        self.mlp_beta_nos = nn.Conv2d(nhidden, 1, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, part_nos, mask=None):


        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        actv_nos = self.mlp_shared_nos(part_nos)
        gamma_nos = self.mlp_gamma_nos(actv_nos)
        beta_nos = self.mlp_beta_nos(actv_nos)

        normalized = self.first_norm(x)

        if mask is not None:
            mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')
            gamma_final = gamma * mask + gamma_nos * (1 - mask)
            beta_final = beta * mask + beta_nos * (1 - mask)
            out = normalized * (1 + gamma_final) + beta_final
        else:
            out = normalized * (1 + gamma_nos) + beta_nos

        return out

class DepthConv(nn.Module):
    def __init__(self, fmiddle, kw=3, padding=1, stride=1):
        super().__init__()

        self.kw = kw
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(self.kw, self.kw), dilation=1, padding=1, stride=stride)
        if True:
            BNFunc = nn.SyncBatchNorm
        else:
            BNFunc = nn.BatchNorm2d

        self.norm_layer = BNFunc(fmiddle, affine=True)

    def forward(self, x, conv_weights):

        N, C, H, W = x.size()

        conv_weights = conv_weights.view(N * C, self.kw * self.kw, H // self.stride, W // self.stride)
        x = self.unfold(x).view(N * C, self.kw * self.kw, H // self.stride, W // self.stride)
        x = torch.mul(conv_weights, x).sum(dim=1, keepdim=False).view(N, C, H // self.stride, W // self.stride)

        return x

class SPADEConcat(nn.Module):
    def __init__(self, opt, norm_nc, label_nc, part_nc):
        super().__init__()

        self.Spade = SPADE(opt, norm_nc, label_nc + part_nc)

    def forward(self, x, seg, part):
        cat = torch.cat((seg, part), 1)
        return self.Spade(x, cat)


class SEAN(nn.Module):
    def __init__(self, opt, norm_nc, label_nc, part_nc):
        super().__init__()

        self.Spade = SPADE(opt, norm_nc, label_nc)

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = opt.part_nc
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(part_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        #
        # sp_norm = get_spectral_norm(opt)
        # self.conv_0 = sp_norm(nn.Conv2d(part_nc, part_nc, kernel_size=3, padding=1))
        # self.conv_1 = sp_norm(nn.Conv2d(part_nc, part_nc, kernel_size=3, padding=1))
        # self.norm_0 = SPADE(opt, part_nc, label_nc)
        # self.norm_1 = SPADE(opt, part_nc, label_nc)
        # self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, part):

        normalized = self.first_norm(x)

        segmap = seg
        partmap = F.interpolate(part, size=x.size()[2:], mode='nearest')


        # partmap = self.activ(self.norm_0(self.conv_0(partmap), seg))
        # partmap = self.activ(self.norm_1(self.conv_1(partmap), seg))

        partmap = self.mlp_shared(partmap)
        gamma_avg = self.mlp_gamma(partmap)
        beta_avg = self.mlp_beta(partmap)

        gamma_spade, beta_spade = self.Spade(x, segmap, return_param=True)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final

        return out

class SAFM(nn.Module):
    def __init__(self, opt, norm_nc, label_nc, part_nc):
        super().__init__()

        self.first_norm = get_norm_layer(opt, norm_nc)
        self.label_nc = label_nc

        ks = opt.spade_ks

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        pw = ks // 2

        self.pre_seg = nn.Sequential(
            nn.Conv2d(label_nc, part_nc, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.pre_dis = nn.Sequential(
            nn.Conv2d(part_nc, part_nc, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gen_weights1 = nn.Sequential(
            nn.Conv2d(part_nc, part_nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(part_nc, part_nc * 9, kernel_size=3, padding=1))

        self.gen_weights2 = nn.Sequential(
            nn.Conv2d(part_nc, part_nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(part_nc, part_nc * 9, kernel_size=3, padding=1))

        self.depconv1 = DepthConv(part_nc)
        self.depconv2 = DepthConv(part_nc)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc + part_nc * 2, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, part):

        # Part 1. generate parameter-free normalized activations
        normalized = self.first_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        part = F.interpolate(part, size=x.size()[2:], mode='nearest')

        pure_seg = segmap
        pure_part = part

        pre_seg_rst = self.pre_seg(pure_seg)
        pre_part_rst = self.pre_dis(pure_part)
        seg_weights1 = self.gen_weights1(pre_seg_rst)
        seg_weights2 = self.gen_weights2(pre_seg_rst)
        dcov_dis1 = self.depconv1(pre_part_rst, seg_weights1)
        dcov_dis2 = self.depconv2(dcov_dis1, seg_weights2)
        dcov_dis_final = torch.cat((pre_part_rst, dcov_dis2), dim=1)

        segmap = torch.cat((pure_seg, dcov_dis_final), dim=1)

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)