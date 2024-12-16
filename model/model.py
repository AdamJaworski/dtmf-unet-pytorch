import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=14):
        # 0 - 11 (w sumie 12) digits + 12 szum, 13 cisza
        super(Model, self).__init__()
        # [bath_size, in_channels, in_data] -> [128, 1, 4406]
        self.conv_down_gr_1_1 = nn.Conv1d(1, 2, kernel_size=7, padding=2) # kernel 5,7 , padding
        self.conv_down_gr_1_2 = nn.Conv1d(2, 2, kernel_size=7, padding=2) # skalowanie w górę np. do 256

        self.conv_down_gr_2_1 = nn.Conv1d(2, 4, kernel_size=7, padding=2)
        self.conv_down_gr_2_2 = nn.Conv1d(4, 4, kernel_size=7, padding=2)

        self.conv_down_gr_3_1 = nn.Conv1d(4, 8, kernel_size=7, padding=2)
        self.conv_down_gr_3_2 = nn.Conv1d(8, 8, kernel_size=7, padding=2)

        self.conv_down_gr_4_1 = nn.Conv1d(8, 16, kernel_size=7, padding=2)
        self.conv_down_gr_4_2 = nn.Conv1d(16, 16, kernel_size=7, padding=2)

        self.conv_down_gr_5_1 = nn.Conv1d(16, 32, kernel_size=7, padding=2)
        self.conv_down_gr_5_2 = nn.Conv1d(32, 32, kernel_size=7, padding=2)

        # self.conv_down_gr_6_1 = nn.Conv1d(32, 64, kernel_size=5, padding=1)
        # self.conv_down_gr_6_2 = nn.Conv1d(64, 64, kernel_size=5, padding=1)

        self.bottleneck1 = nn.Linear(352, 64)
        self.bottleneck2 = nn.Linear(64, 32)
        self.bottleneck3 = nn.Linear(32, 14)
        # dojść do 64 kanałów
        # flatten
        # nn.Linear(32 * 320, 64)
        # nn.Linear(64, 32)
        # nn.Linear(32, 14)
        #self.bottleneck = nn.Linear # ReLU - kilka warstw
        # trzeba zrobić reshape z lin do conv1d


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
        # x = self.pool(x)

        # x = self.relu(self.conv_down_gr_6_1(x))
        # x = self.relu(self.conv_down_gr_6_2(x))

        # Flatten while preserving the batch dimension
        x = x.view(x.size(0), -1)

        x = self.relu(self.bottleneck1(x))
        x = self.relu(self.bottleneck2(x))
        x = self.bottleneck3(x)

        return x# torch.softmax(x, 2) # softmax może być problemem


    # def pool(self, x):
    #     print(x.size())
    #     return self.pool_(x)