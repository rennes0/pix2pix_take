import torch
import torch.nn as nn
import option

#Discriminator
class CNNBlock(nn.Module):
    def __init__(self, input_nc, output_nc, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_nc, output_nc, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(output_nc),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf, device):
        super().__init__()
        self.device = device
        self.initial = nn.Sequential(
            nn.Conv2d(
                input_nc,
                ndf[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        input_nc = ndf[0]
        for feature in ndf[1:]:
            layers.append(
                CNNBlock(input_nc, feature, stride=1 if feature == ndf[-1] else 2),
            )
            input_nc = feature

        layers.append(
            nn.Conv2d(
                input_nc, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y, dis_label, con_label):
        # prepare information
        y_vec_ = torch.cat((dis_label, con_label), 1)
        _, class_num = y_vec_.shape
        batch_size, _, input_size, _ = x.shape
        labels = y_vec_.unsqueeze(2).unsqueeze(3).expand(batch_size, class_num, input_size, input_size)
        labels = labels.to(torch.float32).to(self.device)
        input = torch.cat([x, y, labels], dim=1)
        input = self.initial(input)
        input = self.model(input)
        return input
        


def testD():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(model)
    print(preds.shape)



#Generator
class Block(nn.Module):
    def __init__(self, input_nc, output_nc, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(input_nc, output_nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_nc),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, input_nc, ngf, device):
        self.device = device
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(ngf, ngf * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            ngf * 2, ngf * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            ngf * 4, ngf * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            ngf * 8, ngf * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            ngf * 8, ngf * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            ngf * 8, ngf * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(ngf * 8, ngf * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            ngf * 8 * 2, ngf * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            ngf * 8 * 2, ngf * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            ngf * 8 * 2, ngf * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            ngf * 8 * 2, ngf * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            ngf * 4 * 2, ngf * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(ngf * 2 * 2, ngf, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, 3,  kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, dis_label,con_label):
        y_vec_ = torch.cat((dis_label, con_label), 1)
        _, class_num = y_vec_.shape
        batch_size, _, input_size, _ = x.shape
        labels = y_vec_.unsqueeze(2).unsqueeze(3).expand(batch_size, class_num, input_size, input_size)
        labels = labels.to(torch.float32).to(self.device)
        d1 = self.initial_down(torch.cat([x, labels], 1))
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def testG():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(input_nc=3, ngf=64, device='cuda')
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    #testD()
    testG()