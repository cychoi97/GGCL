import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

from packaging import version


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=8, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=512, conv_dim=32, c_dim=6, repeat_num=7):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class Generator_GGCL(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=8, repeat_num=6):
        super(Generator_GGCL, self).__init__()
        # Init layer
        self.init = nn.Sequential(
            nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Down-sampling layers.
        self.down1 = self.down_layer(conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.down2 = self.down_layer(conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)

        # Bottleneck layers.
        residual_layers = []
        for i in range(repeat_num):
            residual_layers.append(ResidualBlock(dim_in=conv_dim*4, dim_out=conv_dim*4))
        self.residual = nn.Sequential(*residual_layers)

        # Up-sampling layers.
        self.up1 = self.up_layer(conv_dim*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.up2 = self.up_layer(conv_dim*2, kernel_size=3, stride=1, padding=1, bias=False)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(conv_dim, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

        # perceptual
        self.out_feature = nn.Conv2d(conv_dim*2, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def down_layer(self, conv_dim, kernel_size=4, stride=2, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim*2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        return layer

    def up_layer(self, conv_dim, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim//2*4, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        init = self.init(x)
        down1 = self.down1(init)
        down2 = self.down2(down1)
        residual = self.residual(down2)
        up1 = self.up1(residual)
        up2 = self.up2(up1)
        out = self.out(up2)
        out_feature = up1

        return out, out_feature


class Discriminator_GGCL(nn.Module):
    """Discriminator network with U-Net for GGDR."""
    def __init__(self, image_size=512, conv_dim=32, c_dim=6, repeat_num=7):
        super(Discriminator_GGCL, self).__init__()
        # Down-sampling layers
        self.init_down = nn.Sequential(
            spectral_norm(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01)
        )
        self.down1 = self.down_layer(conv_dim, kernel_size=4, stride=2, padding=1)
        self.down2 = self.down_layer(conv_dim*2, kernel_size=4, stride=2, padding=1)
        self.down3 = self.down_layer(conv_dim*4, kernel_size=4, stride=2, padding=1)
        self.down4 = self.down_layer(conv_dim*8, kernel_size=4, stride=2, padding=1)
        self.down5 = self.down_layer(conv_dim*16, kernel_size=4, stride=2, padding=1)
        self.down6 = self.down_layer(conv_dim*32, kernel_size=4, stride=2, padding=1)

        # Up-sampling layers
        self.init_up = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_dim*64, conv_dim*128, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.01)
        )
        self.up1 = self.up_layer(conv_dim*64, kernel_size=1, stride=1, padding=0, bias=False)
        self.up2 = self.up_layer(conv_dim*32, kernel_size=1, stride=1, padding=0, bias=False)
        self.up3 = self.up_layer(conv_dim*16, kernel_size=1, stride=1, padding=0, bias=False)
        self.up4 = self.up_layer(conv_dim*8, kernel_size=1, stride=1, padding=0, bias=False)
        # self.up5 = self.up_layer(conv_dim*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.up5 = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_dim*4, conv_dim*16, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.01)
        )

        kernel_size = int(image_size / np.power(2, repeat_num))

        # Real/fake, class
        self.conv1 = nn.Conv2d(conv_dim*64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(conv_dim*64, c_dim, kernel_size=kernel_size, bias=False)

        # perceptual
        self.out_feature = nn.Conv2d(conv_dim*4, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def down_layer(self, conv_dim, kernel_size=4, stride=2, padding=1):
        layer = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=kernel_size, stride=stride, padding=padding)),
            nn.LeakyReLU(0.01)
        )
        return layer

    def up_layer(self, conv_dim, kernel_size=1, stride=1, padding=0, bias=False):
        layer = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_dim, conv_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.01)
        )
        return layer

    def forward(self, x):
        init_down = self.init_down(x)
        down1 = self.down1(init_down)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)

        init_up = self.init_up(down6)
        up1 = torch.cat([init_up, down5], dim=1)
        up1 = self.up1(up1)
        up2 = torch.cat([up1, down4], dim=1)
        up2 = self.up2(up2)
        up3 = torch.cat([up2, down3], dim=1)
        up3 = self.up3(up3)
        up4 = torch.cat([up3, down2], dim=1)
        up4 = self.up4(up4)
        up5 = torch.cat([up4, down1], dim=1)
        up5 = self.up5(up5)

        out_src = self.conv1(down6)
        out_cls = self.conv2(down6)
        out_feature = up5

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)), out_feature


class PatchNCELoss(nn.Module):
    def __init__(self, batch_size, nce_includes_all_negatives_from_minibatch):
        super().__init__()
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07 # temperature nce

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss