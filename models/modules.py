import torch
import torch.nn as nn
import random


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def get_activ(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('Activation %s not implemented' % name)

def get_norm(name, ch):
    if name == 'bn':
        return nn.BatchNorm2d(ch)
    elif name == 'in':
        return nn.InstanceNorm2d(ch)
    else:
        raise NotImplementedError('Normalization %s not implemented' % name)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc=None, kernel=3, stride=1, activ='lrelu', norm='bn'):
        super(ResidualBlock, self).__init__()

        if outc is None:
            outc = inc // stride

        self.activ = get_activ(activ)
        pad = kernel // 2
        self.input = nn.Conv2d(inc, outc, 1, 1, padding=0)
        self.blocks = nn.Sequential(nn.Conv2d(inc, outc, kernel, 1, pad),
                                    get_norm(norm, outc),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(outc, outc, kernel, 1, pad),
                                    get_norm(norm, outc))

    def forward(self, x):
        return self.activ(self.blocks(x) + self.input(x))


def ConvBlock(inc, outc, ks=3, s=1, p=0, activ='lrelu', norm='in', res=0, resk=3, bn=True):
    conv = nn.Conv2d(inc, outc, ks, s, p)
    blocks = [conv]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, kernel=resk, norm=norm))
    return nn.Sequential(*blocks)

def DeConvBlock(inc, outc, ks=3, s=1, p=0, op=0, activ='relu', norm='in', res=0, resk=3, bn=True):
    deconv = nn.ConvTranspose2d(inc, outc, ks, s, p, op)
    blocks = [deconv]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, norm=norm, kernel=resk))
    return nn.Sequential(*blocks)


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activ='relu', norm='bn'):
        super(LinearBlock, self).__init__()
        # initialize fully connected layer
        deconv = nn.Linear(input_dim, output_dim)
        blocks = [deconv]
        if activ != 'none':
            blocks.append(get_activ(activ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class CrossAttentionLayer(nn.Module):

    def __init__(self, dim, d_model, nhead=1, ffn=True, n_support=None):
        super().__init__()


        self.ffn = ffn

        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1))
        self.key_conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1))
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1))
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)

        if ffn:
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4),
                                     QuickGELU(),
                                     nn.Linear(d_model * 4, d_model))

        # self.conv_att = ConvBlock(dim, 16, 3, 1, 1)
        # self.att = nn.Sequential(ConvBlock(dim + 16 * (n_support), 32, 3, 1, 1),
        #                          nn.Conv2d(32, n_support, 3, 1, 1),
        #                          nn.Softmax(dim=1))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, supports, image_mask_split=True):

        m_batchsize, C, height, width = query.size()

        query_view = query.view(m_batchsize, -1, width * height).permute(2, 0, 1)
        proj_query = self.query_conv(query).view(m_batchsize, -1, width * height).permute(2, 0, 1)

        proj_keys = []
        proj_values = []

        for supp in supports:
            if image_mask_split:
                supp_img, supp_mask = supp[0], supp[1]
            else:
                supp_img, supp_mask = supp, supp

            proj_key = self.key_conv(supp_img).view(m_batchsize, -1, width * height).permute(2, 0, 1)
            proj_value = self.value_conv(supp_mask).view(m_batchsize, -1, width * height).permute(2, 0, 1)
            proj_keys.append(proj_key)
            proj_values.append(proj_value)

        proj_value = torch.cat(proj_values, dim=0)
        proj_key = torch.cat(proj_keys, dim=0)

        query_att = self.multihead_attn(query=proj_query,
                                      key=proj_key,
                                      value=proj_value)[0]
        query_att = query_view + self.ln1(query_att)

        if self.ffn:
            query_att = query_att + self.mlp(self.ln2(query_att))

        query_att = query_att.view(height, width, m_batchsize, C).permute(2, 3, 0, 1)

        # outs = []
        # outs_att = []
        # for supp in supports:
        #     if image_mask_split:
        #         supp_img, supp_mask = supp[0], supp[1]
        #     else:
        #         supp_img, supp_mask = supp, supp
        #
        #     proj_key = self.key_conv(supp_img).view(m_batchsize, -1, width * height).permute(2, 0, 1)
        #     proj_value = self.value_conv(supp_mask).view(m_batchsize, -1, width * height).permute(2, 0, 1)
        #
        #     query_att = self.multihead_attn(query=proj_query,
        #                                     key=proj_key,
        #                                     value=proj_value)[0]
        #     query_att = query_view + self.ln1(query_att)
        #
        #     if self.ffn:
        #         query_att = query_att + self.mlp(self.ln2(query_att))
        #
        #     query_att = query_att.view(height, width, m_batchsize, C).permute(2, 3, 0, 1)
        #     outs.append(query_att)
        #     outs_att.append(self.conv_att(query_att))
        #
        #
        # att = self.att(torch.cat([query] + outs_att, 1))
        # query_att = outs[0] * att[:, 0:1]
        # for i in range(1, len(outs)-1):
        #     query_att += outs[i] * att[:, i:i+1]

        return query_att