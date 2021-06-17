import torch
import torch.nn as nn
import torch.nn.functional as F
from util import N2_Normalization, Recover_from_density

class reg_preTrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(reg_preTrain, self).__init__()

        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc = self.conv_pix(x)
        enc = self.bn(enc)

        B, C, W, H = enc.shape
        enc = enc.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
        enc = self.linear(enc)

        return enc


class tc_preTrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(tc_preTrain, self).__init__()

        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = 2.0

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

    def forward(self, x):
        enc = self.conv_tc(x)
        enc = self.bn(enc)

        out = self.linear(self.AvgPool(enc).squeeze())

        return self.normalize(out) * self.alpha


class inference_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=8, img_height=8):
        super(inference_net, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.BatchNorm2d(base_channels * 16),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc = self.conv1(x)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out


class UrbanSTC(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32, extend=1, ext_flag=False):
        super(UrbanSTC, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.ext_flag = ext_flag

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # reg_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        if self.ext_flag == True:
            self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, img_width * img_height),
                nn.ReLU(inplace=True)
            )

        self.conv_combine = nn.Sequential(
            nn.Conv2d(base_channels*3, base_channels*extend, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        for i in range(2):
            upsampling += [nn.Conv2d(base_channels*extend, base_channels*extend * 4, 3, 1, 1),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels*extend, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ext):

        inp = x

        if self.ext_flag == True:
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]

            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)

            inp = torch.add(x, ext_out)

        enc_inf = self.conv1(inp)
        enc_tc = self.conv_tc(inp)
        enc_pix = self.conv_pix(inp)

        enc = torch.cat((enc_inf, enc_tc, enc_pix), dim=1)
        enc = self.conv_combine(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out