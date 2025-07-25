import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class SAM2(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM2, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x):
        x1 = self.conv1(x)
        img = self.conv2(x)
        # x2 = torch.sigmoid(self.conv3(img))
        # x1 = x1*x2
        # x1 = x1+x
        return x1, img

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level4 = [CAB(n_feat+(scale_unetfeats*3), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level4 = nn.Sequential(*self.encoder_level4)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)
        self.down34  = DownSample(n_feat+2*scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        # x = self.down34(enc3)
        # enc4 = self.encoder_level4(x)
        
        return [enc1, enc2, enc3]

class Encoder_f(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder_f, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, x_2, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        enc11 = self.encoder_level1(x_2)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)
        x_2 = self.down12(enc11)

        enc2 = self.encoder_level2(x)
        enc22 = self.encoder_level2(x_2)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)
        x_2 = self.down23(enc22)

        enc3 = self.encoder_level3(x)
        enc33 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level4 = [CAB(n_feat+(scale_unetfeats*3), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level4 = nn.Sequential(*self.decoder_level4)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn3 = CAB(n_feat+scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)
        self.up43  = SkipUpSample(n_feat+scale_unetfeats*2, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs

        # dec4 = self.decoder_level4(enc4)

        # x = self.up43(dec4, self.skip_attn3(enc3))

        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

class Decoder_f(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder_f, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

        self.convd11 = nn.Sequential(
                nn.Conv2d((n_feat)*2, (n_feat), 1, padding=0, bias=bias),
                nn.ReLU(inplace=True)
        )

        self.convd22 = nn.Sequential(
                nn.Conv2d((n_feat+(scale_unetfeats))*2, (n_feat+(scale_unetfeats)), 1, padding=0, bias=bias),
                nn.ReLU(inplace=True)
        )

        self.convd33 = nn.Sequential(
                nn.Conv2d((n_feat+(scale_unetfeats*2))*2, (n_feat+(scale_unetfeats*2)), 1, padding=0, bias=bias),
                nn.ReLU(inplace=True)
        )

    def forward(self, outs, conf):
        enc1, enc2, enc3 = outs
        conf_1, conf_2, conf_3 = conf

        # enc1 = enc1 + enc11
        # enc2 = enc2 + enc21
        # enc3 = enc3 + enc31
        # enc1 = torch.cat((enc1, enc21), 1)
        # enc1 = self.convd22(enc1)

        # enc3 = torch.cat((enc3, enc31), 1)
        # enc3 = self.convd33(enc3)
        # dec3 = self.decoder_level3(enc3)

        # enc2 = torch.cat((enc2, enc21), 1)
        # enc2 = self.convd22(enc2)

        # x = self.up32(dec3, self.skip_attn2(enc2))
        # dec2 = self.decoder_level2(x)

        # enc1 = torch.cat((enc1, enc11), 1)
        # enc1 = self.convd11(enc1)

        # x = self.up21(dec2, self.skip_attn1(enc1))
        # dec1 = self.decoder_level1(x)

        enc3 = torch.mul(enc3, conf_1)
        dec3 = self.decoder_level3(enc3)
        # dec3 = torch.cat((dec3, enc31), 1)
        # dec3 = self.convd33(dec3)

        enc2 = torch.mul(enc2, conf_2)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        # dec2 = torch.cat((dec2, enc21), 1)
        # dec2 = self.convd22(dec2)

        enc1 = torch.mul(enc1, conf_3)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        # dec1 = torch.cat((dec1, enc11), 1)
        # dec1 = self.convd11(dec1)

        return [dec1,dec2,dec3]

class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class DownSample_2(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample_2, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class map_conv(nn.Module):
    
    def __init__(self, input_channel, output_channel, track_running_static=True):
        super(map_conv, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.track = track_running_static
        filters = [64, 64, 64, 64]

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channel, filters[0], 3, 1, 1),
                                  nn.InstanceNorm2d(filters[0], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[0], filters[1], 3, 1, 1),
                                  nn.InstanceNorm2d(filters[1], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[1], filters[2], 1),
                                #   nn.InstanceNorm2d(filters[1], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[2], filters[3], 1),
                                #   nn.InstanceNorm2d(filters[1], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[3], self.output_channel, 1),)

    def forward(self, input):

        output = self.conv1(input)

        return output

class atrous_conv(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(atrous_conv, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x


def flat_conv(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)


class _NetG(nn.Module):
    def __init__(self, in_c=6, out_c=1, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(_NetG, self).__init__()

        act=nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(4, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(1, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(1, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder_f(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)


        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam22 = SAM2(n_feat, kernel_size=1, bias=bias)
        self.model_map = map_conv(2, 1)
        self.model_map2 = map_conv(80, 1)

        self.down1  = DownSample_2(1, 0)
        self.down2  = DownSample_2(1, 0)

        self.convd11 = nn.Sequential(
                nn.Conv2d(n_feat*3, n_feat, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True)
        )

        self.convd22 = nn.Sequential(
                nn.Conv2d(16*3, 1, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True)
        )

        self.aspp1 = atrous_conv(1, 16, 1)
        self.aspp2 = atrous_conv(1, 16, 2)
        self.aspp3 = atrous_conv(1, 16, 3)

        self.norm = F.instance_norm
        self.relu = nn.ReLU(inplace=True)
        self.F1 = flat_conv(6, 64)

    def forward(self, x3_img):
        corr_image1 = x3_img[:,0:4,:,:]
        # corr_image2 = x3_img[:,2:4,:,:]
        depth = x3_img[:,4,:,:].unsqueeze(1)
        amp = x3_img[:,5,:,:].unsqueeze(1)
        # d_a = x3_img[:,4:6,:,:]
        conf = torch.where(amp < 0.002, 0, 1)

        # amp = self.norm(self.relu(torch.cat([self.aspp1(amp), self.aspp2(amp), self.aspp3(amp)], dim=1)))
        # amp = self.convd22(amp)
        # x3_img = torch.cat((x3_img, amp), 1)

        # x = self.F1(x3_img)
        # x = self.norm(self.relu(self.F2(torch.cat([x, amp], dim=1))))
        # conf_3 = self.convd22(amp)
        # conf_2 = self.down1(conf_3)
        # conf_1 = self.down2(conf_2)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        fea1= self.shallow_feat1(corr_image1)
        fea2= self.shallow_feat2(depth*amp)
        fea3= self.shallow_feat3(depth)
        fea2= torch.mul(fea2, conf)


        fea = torch.cat((fea1, fea2, fea3), 1)
        fea = self.convd11(fea)

        # fea= self.shallow_feat3(x)
        
        ## Process features of all 4 patches with Encoder of Stage 1
        fea1_enc = self.stage2_encoder(fea)
        # fea2_enc = self.stage2_encoder(fea2)

        self.feature = fea1_enc
        
        ## Pass features through Decoder of Stage 1
        fea1_dec= self.stage2_decoder(fea1_enc)
        # fea1_dec= self.stage1_decoder(fea1_enc, [conf_1, conf_2, conf_3])

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img = self.sam12(fea1_dec[0], depth)


        return stage1_img


class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 128 x 128
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),            
            #nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 64 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),            
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        
            # state size. (128) x 32 x 32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(7680, 64) #7680, 2048 #1024
        self.fc2 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)
        # state size. (512) x 6 x 6
        out = out.view(out.size(0), -1)

        # state size. (512 x 6 x 6)
        # print("=======================", out.shape)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        #out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)

class _NetDedge(nn.Module):
    def __init__(self):
        super(_NetDedge, self).__init__()

        self.features = nn.Sequential(

            # input is (2) x 128 x 128
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),            
            #nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 64 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),            
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        
            # state size. (128) x 32 x 32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(1024, 64) #2048
        self.fc2 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)
        # state size. (512) x 6 x 6
        out = out.view(out.size(0), -1)

        # state size. (512 x 6 x 6)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        #out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)


'''
class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 96 x 96
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),            
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),            
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (64) x 48 x 48
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        
            # state size. (128) x 48 x 48
            #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 24 x 24
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 12 x 12
            #nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 12 x 12
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input, label):

        out = torch.cat([input, label], 1)
        out = self.features(out)
        # state size. (512) x 6 x 6
        out = out.view(out.size(0), -1)

        # state size. (512 x 6 x 6)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        #out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)
'''
