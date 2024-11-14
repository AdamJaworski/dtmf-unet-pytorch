import sys

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=14):
        # 0 - 11 (w sumie 12) digits + 12 szum, 13 cisza
        super(Model, self).__init__()

        self.conv_down_gr_1_1 = nn.Conv1d(1, 8, kernel_size=7, padding=2) # kernel 5,7 , padding
        self.conv_down_gr_1_2 = nn.Conv1d(8, 8, kernel_size=7, padding=2) # skalowanie w górę np. do 256

        self.conv_down_gr_2_1 = nn.Conv1d(8, 16, kernel_size=7, padding=2)
        self.conv_down_gr_2_2 = nn.Conv1d(16, 16, kernel_size=7, padding=2)

        self.conv_down_gr_3_1 = nn.Conv1d(16, 32, kernel_size=7, padding=2)
        self.conv_down_gr_3_2 = nn.Conv1d(32, 32, kernel_size=7, padding=2)

        self.conv_down_gr_4_1 = nn.Conv1d(32, 64, kernel_size=7, padding=2)
        self.conv_down_gr_4_2 = nn.Conv1d(64, 64, kernel_size=7, padding=2)

        self.conv_down_gr_5_1 = nn.Conv1d(64, 128, kernel_size=7, padding=2)
        self.conv_down_gr_5_2 = nn.Conv1d(128, 128, kernel_size=7, padding=2)

        self.conv_down_gr_6_1 = nn.Conv1d(128, 256, kernel_size=7, padding=2)
        self.conv_down_gr_6_2 = nn.Conv1d(256, 256, kernel_size=7, padding=2)

        self.bottleneck = nn.Conv1d(256, 256, kernel_size=7, padding=2)

        self.conv_up_gr_1_1 = nn.Conv1d(256, 256, kernel_size=5, padding=2)

        self.upconv1 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, output_padding=1)

        self.conv_up_gr_2_1 = nn.Conv1d(128, 128, kernel_size=5, padding=2)

        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, output_padding=1)

        self.conv_up_gr_3_1 = nn.Conv1d(64, 64, kernel_size=5, padding=2)

        self.output_conv = nn.Conv1d(64, num_classes, kernel_size=5, padding=2)

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
        x = self.pool(x)

        x = self.relu(self.conv_down_gr_6_1(x))
        x = self.relu(self.conv_down_gr_6_2(x))
        x = self.bottleneck(x)

        x = self.conv_up_gr_1_1(x)
        x = self.upconv1(x)

        x = self.conv_up_gr_2_1(x)
        x = self.upconv2(x)

        x = self.conv_up_gr_3_1(x)
        x = self.output_conv(x)

        return x# torch.softmax(x, 2) # softmax może być problemem
