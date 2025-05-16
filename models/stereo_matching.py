import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import BasicConv, Conv2x, convbn_2d_lrelu, convbn_2d_Tanh, build_gwc_volume, build_concat_volume, disparity_regression
import math

chans = [32, 48, 64, 96, 160]

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()

        self.block_2 = nn.Sequential(
                      BasicConv(3, chans[0], kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(chans[0], chans[0], 3, 1, 1, bias=False),
                      nn.BatchNorm2d(chans[0]), nn.ReLU())
        self.block_4 = nn.Sequential(
                      BasicConv(chans[0], chans[1], kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(chans[1], chans[1], 3, 1, 1, bias=False),
                      nn.BatchNorm2d(chans[1]), nn.ReLU())
        self.block_8 = nn.Sequential(
                      BasicConv(chans[1], chans[2], kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(chans[2], chans[2], 3, 1, 1, bias=False),
                      nn.BatchNorm2d(chans[2]), nn.ReLU())
        self.block_16 = nn.Sequential(
                      BasicConv(chans[2], chans[3], kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(chans[3], chans[3], 3, 1, 1, bias=False),
                      nn.BatchNorm2d(chans[3]), nn.ReLU())

        self.deconv_16_8 = Conv2x(chans[3], chans[2], deconv=True, concat=True, keep_concat=False)
        self.deconv_8_4 = Conv2x(chans[2], chans[1], deconv=True, concat=True, keep_concat=False)
        self.deconv_4_2 = Conv2x(chans[1], chans[0], deconv=True, concat=True, keep_concat=False)

        self.conv_2_4 = Conv2x(chans[0], chans[1], concat=True, keep_concat=False)
        self.conv_4_8 = Conv2x(chans[1], chans[2], concat=True, keep_concat=False)
        self.conv_8_16 = Conv2x(chans[2], chans[3], concat=True, keep_concat=False)

        self.weight_init()

    def forward(self, x):
        x2 = self.block_2(x)
        x4 = self.block_4(x2)
        x8 = self.block_8(x4)
        x16 = self.block_16(x8)

        x8 = self.deconv_16_8(x16, x8)
        x4 = self.deconv_8_4(x8, x4)
        x2 = self.deconv_4_2(x4, x2)

        x4 = self.conv_2_4(x2, x4)
        x8 = self.conv_4_8(x4, x8)
        x16 = self.conv_8_16(x8, x16)

        return x2, [x4, x8, x16]


class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, 32, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.last_for_guidance = BasicConv(32, 32, is_3d=True, bn=False, relu=False, kernel_size=3,
                  padding=1, stride=1, dilation=1)


        self.agg = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.feature_att_8 = channelAtt(in_channels * 2, chans[2])
        self.feature_att_16 = channelAtt(in_channels * 4, chans[3])
        self.feature_att_up_8 = channelAtt(in_channels * 2, chans[2])

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)
        conv = self.last_for_guidance(conv).permute(0, 2, 1, 3, 4).contiguous()  # [B,G,D,H,W] -> [B,D,G,H,W]
        return conv


class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv_final = BasicConv(in_channels, 1, is_3d=True, bn=False, relu=False, kernel_size=3,
                                    padding=1, stride=1, dilation=1)

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.feature_att_8 = channelAtt(in_channels * 2, chans[2])
        self.feature_att_16 = channelAtt(in_channels * 4, chans[3])
        self.feature_att_up_8 = channelAtt(in_channels * 2, chans[2])

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)
        conv = self.conv_final(conv)

        return conv


class StereoMatching(nn.Module):
    def __init__(self, maxdisp, att_weights_only=False):
        super(StereoMatching, self).__init__()
        self.maxdisp = maxdisp
        self.att_weights_only = att_weights_only
        self.feature = Feature()

        self.spx_4 = nn.Sequential(
            BasicConv(chans[1], chans[1], kernel_size=3, stride=1, padding=1),
            BasicConv(chans[1], chans[1], kernel_size=3, stride=1, padding=1),)
        self.spx_2 = Conv2x(chans[1], chans[0], True)
        self.spx = BasicConv(chans[0]*2, 32, deconv=True, is_3d=False, bn=True,
                                  relu=True, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2))
        self.guide_conv1 = convbn_2d_lrelu(32, 16, 1, 1, 0)
        self.guide_conv2 = convbn_2d_Tanh(16, 1, 1, 1, 0)

        self.conv = BasicConv(chans[1], chans[1], kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(chans[1], chans[1], kernel_size=1, padding=0, stride=1)
        self.patch = nn.Conv3d(12, 12, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=12, padding=(0, 1, 1),
                               bias=False)
        self.corr_feature_att_4 = channelAtt(12, chans[1])
        self.hourglass_att = hourglass_att(12)

        self.concat_feature = nn.Sequential(
            BasicConv(chans[1], 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False))
        self.concat_stem = BasicConv(32, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.concat_feature_att_4 = channelAtt(16, chans[1])
        self.hourglass = hourglass(16)


    def BG_upsample(self, volume, guide):
        B, _, H, W = guide.shape
        dtype = guide.dtype
        device = guide.device

        hg, wg = torch.meshgrid([torch.arange(0.5, H+0.5, 1, dtype=dtype, device=device), torch.arange(0.5, W+0.5, 1, dtype=dtype, device=device)], indexing='ij')  # HxW
        hg = hg.repeat(B, 1, 1).unsqueeze(3) / H * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.repeat(B, 1, 1).unsqueeze(3) / W * 2 - 1  # norm to [-1,1] NxHxWx1

        guide = guide.permute(0, 2, 3, 1).contiguous()  # [B,1,H,W] -> [B,H,W,1]
        guidemap = torch.cat([wg, hg, guide], dim=3).unsqueeze(1)  # Bx1xHxWx3


        slice_dict = F.grid_sample(volume, guidemap, align_corners=False) # [B,d,G,h,w] -> [B,d,1,H,W]

        wa = torch.arange(1, 0, -0.25, dtype=dtype, device=device).view(1,1,-1,1,1)
        wb = torch.arange(0, 1, 0.25, dtype=dtype, device=device).view(1,1,-1,1,1)
        volume_up = wa * slice_dict[:,:-1] + wb * slice_dict[:,1:]

        return volume_up.view(B, -1, H, W)  # [B,D,H,W]


    def forward(self, left, right):

        stem_2x, features_left = self.feature(left)
        _, features_right = self.feature(right)


        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        corr_volume = build_gwc_volume(match_left, match_right, self.maxdisp//4, 12)
        corr_volume = self.patch(corr_volume)
        cost_att = self.corr_feature_att_4(corr_volume, features_left[0])
        att_weights = self.hourglass_att(cost_att, features_left)
        att_weights_prob = F.softmax(att_weights, dim=2)

        att_weights_up = F.interpolate(att_weights, size=[self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        att_weights_prob_up = F.softmax(att_weights_up, dim=2)
        pred_att_up = disparity_regression(att_weights_prob_up.squeeze(1), self.maxdisp)

        if self.att_weights_only:
            return pred_att_up, pred_att_up, torch.ones_like(pred_att_up).unsqueeze(1)

        concat_features_left = self.concat_feature(features_left[0])
        concat_features_right = self.concat_feature(features_right[0])
        concat_volume = build_concat_volume(concat_features_left, concat_features_right, self.maxdisp // 4)
        volume = att_weights_prob * concat_volume

        volume = self.concat_stem(volume)
        volume = self.concat_feature_att_4(volume, features_left[0])
        cost = self.hourglass(volume, features_left)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        xspx = self.spx(xspx)
        guide = self.guide_conv2(self.guide_conv1(xspx))

        cost_up = self.BG_upsample(cost, guide)
        prob_up = F.softmax(cost_up, dim=1)
        pred_up = disparity_regression(prob_up, self.maxdisp-4)

        pred_idx = torch.floor(pred_up).long()
        pred_idx_1 = torch.clamp(pred_idx + 1, min=0, max=self.maxdisp - 5)
        pred_conf = torch.gather(prob_up, dim=1, index=pred_idx.unsqueeze(1)) + torch.gather(prob_up, dim=1, index=pred_idx_1.unsqueeze(1))

        return pred_up, pred_att_up, pred_conf
