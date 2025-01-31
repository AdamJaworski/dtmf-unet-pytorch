import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=14):
        # 0 - 11 (w sumie 12) digits + 12 szum, 13 cisza
        super(Model, self).__init__()
        # [bath_size, in_channels, in_data] -> [128, 1, 4406]
        self.conv_down_gr_1_1 = nn.Conv1d(1, 7,  kernel_size=7, padding=2) # kernel 5,7 , padding
        self.conv_down_gr_1_2 = nn.Conv1d(7, 14, kernel_size=7, padding=2) # skalowanie w górę np. do 256

        self.conv_down_gr_2_1 = nn.Conv1d(14, 28, kernel_size=7, padding=2)
        self.conv_down_gr_2_2 = nn.Conv1d(28, 56, kernel_size=7, padding=2)

        self.conv_down_gr_3_1 = nn.Conv1d(56, 112, kernel_size=7, padding=2)
        self.conv_down_gr_3_2 = nn.Conv1d(112, 224, kernel_size=7, padding=2)

        self.conv_down_gr_4_1 = nn.Conv1d(224, 286, kernel_size=7, padding=2)
        self.conv_down_gr_4_2 = nn.Conv1d(286, 324, kernel_size=7, padding=2)

        self.conv_down_gr_5_1 = nn.Conv1d(324, 368, kernel_size=7, padding=2)
        self.conv_down_gr_5_2 = nn.Conv1d(368, 396, kernel_size=7, padding=2)


        self.bottleneck1 = nn.Linear(4356, 24)
        self.bottleneck2 = nn.Linear(24, 14)

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

        # Flatten while preserving the batch dimension
        x = x.view(x.size(0), -1)
        x = self.relu(self.bottleneck1(x))
        x = self.bottleneck2(x)

        return x# torch.softmax(x, 2) # softmax może być problemem


    # def pool(self, x):
    #     print(x.size())
    #     return self.pool_(x)