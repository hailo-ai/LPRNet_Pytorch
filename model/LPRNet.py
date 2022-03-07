import torch.nn as nn
import torch


class res_block(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, ks=3, downsample=None, padding=1):
        super(res_block, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=ks, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=ch_out),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act(out)
        return out


class downsample(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super(downsample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, device, drop=False):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.device = device       

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            res_block(ch_in=64, ch_out=64, padding=1),
            res_block(ch_in=64, ch_out=128, padding=1,
                      downsample=downsample(64, 128, kernel_size=1, stride=1)),

            # s2
            res_block(ch_in=128, ch_out=128, stride=2, padding=1,
                      downsample=downsample(128, 128, kernel_size=1, stride=2)),
            res_block(ch_in=128, ch_out=256, padding=1,
                      downsample=downsample(128, 256, kernel_size=1, stride=1)),
            ) # (38 x 150)
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256)
            )

        self.stage2 = nn.Sequential(
            res_block(ch_in=256, ch_out=256, stride=2, padding=1,
                      downsample=downsample(256, 256, kernel_size=1, stride=2)),
            res_block(ch_in=256, ch_out=256, padding=1)
            ) # (19 x 75)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            )

        self.stage3 = nn.Sequential(
            res_block(ch_in=256, ch_out=256, stride=2, padding=1,
                      downsample=downsample(256, 256, kernel_size=1, stride=2)),
            res_block(ch_in=256, ch_out=256, stride=2, padding=1,
                      downsample=downsample(256, 256, kernel_size=1, stride=2))
            ) # (5 x 19)
        if drop:
            self.stage4 = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=1, padding=(0, 2)),  # (6 x 24)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(5, 1), stride=1, padding=(2, 0)), # (6 x 24)
                nn.BatchNorm2d(num_features=class_num),
                nn.ReLU(),
            )    
        else:
            self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=1, padding=(0, 2)),  # (6 x 24)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(5, 1), stride=1, padding=(2, 0)), # (6 x 24)
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
            ) # (5 x 19)

        self.bn = nn.Sequential(
            nn.BatchNorm2d(num_features=256)
        )
        self.bn4 = nn.Sequential(
            nn.BatchNorm2d(num_features=self.class_num)
        )

        self.container = nn.Sequential(
            nn.Conv2d(in_channels=768+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        
        skip1 = self.downsample1(stage1)
        skip2 = self.downsample2(stage2)
        skip3 = stage3
        skip4 = stage4

        x = torch.cat([skip1, skip2, skip3, skip4], 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

def build_lprnet(lpr_max_len=8, phase_train=False, class_num=66, dropout_rate=0.5, device=torch.device("cuda:0"), drop=False):

    Net = LPRNet(lpr_max_len, phase_train, class_num, dropout_rate, device, drop=drop)

    if phase_train:
        return Net.train()
    else:
        return Net.eval()
