import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=12):
        super(Model, self).__init__()

        self.conv_down_gr_1_1 = nn.Conv1d(1, 16, kernel_size=7, padding=1) # kernel 5,7 , padding
        self.conv_down_gr_1_2 = nn.Conv1d(16, 16, kernel_size=7, padding=1) # skalowanie w górę np. do 256

        self.conv_down_gr_2_1 = nn.Conv1d(16, 32, kernel_size=7, padding=1)
        self.conv_down_gr_2_2 = nn.Conv1d(32, 32, kernel_size=7, padding=1)

        self.conv_down_gr_3_1 = nn.Conv1d(32, 64, kernel_size=7, padding=1)
        self.conv_down_gr_3_2 = nn.Conv1d(64, 64, kernel_size=7, padding=1)

        self.conv_down_gr_4_1 = nn.Conv1d(64, 128, kernel_size=7, padding=1)
        self.conv_down_gr_4_2 = nn.Conv1d(128, 128, kernel_size=7, padding=1)

        self.conv_down_gr_5_1 = nn.Conv1d(128, 256, kernel_size=7, padding=1)
        self.conv_down_gr_5_2 = nn.Conv1d(256, 256, kernel_size=7, padding=1)

        self.bottleneck = nn.Conv1d(256, 256, kernel_size=7, padding=1)

        self.conv_up_gr_1_1 = nn.Conv1d(256, 256, kernel_size=7, padding=1)

        self.upconv1 = nn.ConvTranspose1d(256, 128, kernel_size=7, stride=2, output_padding=0)

        self.conv_up_gr_2_1 = nn.Conv1d(128, 128, kernel_size=7, padding=1)

        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, output_padding=0)

        self.conv_up_gr_3_1 = nn.Conv1d(64, 64, kernel_size=7, padding=1)

        self.output_conv = nn.Conv1d(64, num_classes, kernel_size=7)

        self.pool = nn.MaxPool1d(4) # MaxPool (4)
        self.relu = nn.ReLU()
        # Bottleneck!!!


    def forward(self, x):
        x = self.relu(self.conv_down_gr_1_1(x))
        x = self.relu(self.conv_down_gr_1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv_down_gr_2_1(x))
        x = self.relu(self.conv_down_gr_2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv_down_gr_3_1(x))
        x = self.relu(self.conv_down_gr_3_2(x))
        x = self.pool(x)

        x = self.relu(self.conv_down_gr_4_1(x))
        x = self.relu(self.conv_down_gr_4_2(x))
        x = self.pool(x)

        x = self.relu(self.conv_down_gr_5_1(x))
        x = self.relu(self.conv_down_gr_5_2(x))

        x = self.bottleneck(x)

        x = self.conv_up_gr_1_1(x)
        x = self.upconv1(x)

        x = self.conv_up_gr_2_1(x)
        x = self.upconv2(x)

        x = self.conv_up_gr_3_1(x)
        x = self.output_conv(x)

        print(x.size())
        return torch.softmax(x, 1)
