from mindspore import nn
import mindspore.numpy as np

# UNet模型


class double_conv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.double_conv = nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),
                                             nn.BatchNorm2d(out_ch), nn.ReLU(),
                                             nn.Conv2d(out_ch, out_ch, 3),
                                             nn.BatchNorm2d(out_ch), nn.ReLU())

    def construct(self, x):
        x = self.double_conv(x)
        return x
class UNet(nn.Cell):
    def __init__(self, in_ch=3):
        super(UNet, self).__init__()
        # Encoder
        # [N,3,256,256]->[N,64,256,256]
        self.double_conv1 = double_conv(in_ch, 64)
        # [N,64,256,256]->[N,64,128,128]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,64,128,128]->[N,128,128,128]
        self.double_conv2 = double_conv(64, 128)
        # [N,128,128,128]->[N,128,64,64]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,128,64,64]->[N,256,64,64]
        self.double_conv3 = double_conv(128, 256)
        # [N,256,64,64]->[N,256,32,32]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,256,32,32]->[N,512,32,32]
        self.double_conv4 = double_conv(256, 512)
        # [N,512,32,32]->[N,512,16,16]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,512,16,16]->[N,1024,16,16]
        self.double_conv5 = double_conv(512, 1024)

        # Decoder
        # [N,1024,16,16]->[N,1024,32,32]
        self.upsample1 = nn.ResizeBilinear()
        # [N,1024+512,32,32]->[N,512,32,32]
        self.double_conv6 = double_conv(1024 + 512, 512)
        # [N,512,32,32]->[N,512,64,64]
        self.upsample2 = nn.ResizeBilinear()
        # [N,512+256,64,64]->[N,256,64,64]
        self.double_conv7 = double_conv(512 + 256, 256)
        # [N,256,64,64]->[N,256,128,128]
        self.upsample3 = nn.ResizeBilinear()
        # [N,256+128,128,128]->[N,128,128,128]
        self.double_conv8 = double_conv(256 + 128, 128)
        # [N,128,128,128]->[N,128,256,256]
        self.upsample4 = nn.ResizeBilinear()
        # [N,128+64,256,256]->[N,64,256,256]
        self.double_conv9 = double_conv(128 + 64, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def construct(self, x):
        feature1 = self.double_conv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.double_conv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.double_conv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.double_conv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.double_conv5(tmp)

        up_feature1 = self.upsample1(feature5, scale_factor=2)
        tmp = np.concatenate((feature4, up_feature1), axis=1)
        tmp = self.double_conv6(tmp)
        up_feature2 = self.upsample2(tmp, scale_factor=2)
        tmp = np.concatenate((feature3, up_feature2), axis=1)
        tmp = self.double_conv7(tmp)
        up_feature3 = self.upsample3(tmp, scale_factor=2)
        tmp = np.concatenate((feature2, up_feature3), axis=1)
        tmp = self.double_conv8(tmp)
        up_feature4 = self.upsample4(tmp, scale_factor=2)
        tmp = np.concatenate((feature1, up_feature4), axis=1)
        tmp = self.double_conv9(tmp)
        output = self.final(tmp)
        return output
