import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vggloss import VGG19


class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

    def loss(self, input, label, for_real, ignore_idx=None):
        #--- balancing classes ---
        weight_map = get_class_balancing(self.opt, input, label)
        #--- n+1 loss ---
        target = get_n1_target(self.opt, input, label, for_real)
        if ignore_idx is not None and for_real:
            loss = F.cross_entropy(input, target, reduction='none', ignore_index=ignore_idx)
        else:
            loss = F.cross_entropy(input, target, reduction='none')

        if for_real:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

    def loss_ce(self, input, label, weight=None, for_real=False):

        weight_map = get_class_balancing(self.opt, input, label)
        # weight_map = torch.ones_like(weight_map)
        #--- balancing classes ---
        # loss = F.cross_entropy(input, torch.argmax(label, dim=1).long(), reduction='none', ignore_index=0)
        loss = F.cross_entropy(input, torch.argmax(label, dim=1).long(), reduction='none', ignore_index=0)
        if weight_map is not None:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)

        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map


def get_n1_target(opt, input, label, target_is_real):
    targets = get_target_tensor(opt, input, target_is_real)
    num_of_classes = label.shape[1]
    integers = torch.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(opt, input, target_is_real):
    if opt.gpu_ids != "-1":
        if target_is_real:
            return torch.cuda.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        else:
            return torch.cuda.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)
    else:
        if target_is_real:
            return torch.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        else:
            return torch.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, select_layers=[0,1,2,3,4]):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0

        for i in select_layers:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2 * (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        self.center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.],
                                           [0., 0., 0.]])  # , device=device)

        # TODO: modified this as one conv with 8 channels for efficiency
        self.pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 1.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 1., 0.], [0., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [0., 1., 0.]]),  # , device=device),
            torch.tensor([[1., 0., 0.], [0., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 1.], [0., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [1., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [0., 0., 1.]]),  # , device=device),
        ]

    def forward(self, x):
        device = x.device
        pairwise_loss = []
        n, c, h, w = x.shape
        x_view = x.view(n*c, 1, h, w)
        # Sigmoid transform to [0, 1]
        mask_logits_normalize = x_view.sigmoid()

        # Compute pairwise loss for each col/row MIL
        for w in self.pairwise_weights_list:
            conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding=(1, 1))
            weights = self.center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            conv.weight = torch.nn.Parameter(weights)
            for param in conv.parameters():
                param.requires_grad = False
            aff_map = conv(mask_logits_normalize)

            cur_loss = (aff_map ** 2)
            cur_loss = torch.mean(cur_loss)
            pairwise_loss.append(cur_loss)
        pairwise_loss = torch.mean(torch.stack(pairwise_loss))
        return pairwise_loss
