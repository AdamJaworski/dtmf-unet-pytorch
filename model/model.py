import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=12):
        super(Model, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, output_padding=0)

        self.conv7 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, output_padding=0)  # Added output_padding=1

        self.conv9 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv1d(64, num_classes, kernel_size=1)

        self.pool = nn.MaxPool1d(2)  # Use ceil_mode=True
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1_pooled = self.pool(x1)

        x2 = self.relu(self.conv3(x1_pooled))
        x2 = self.relu(self.conv4(x2))
        x2_pooled = self.pool(x2)

        x3 = self.relu(self.conv5(x2_pooled))
        x3 = self.relu(self.conv6(x3))

        x_up1 = self.upconv1(x3)
        if x_up1.size() == x2.size():
            x_up1 = torch.cat([x_up1, x2], dim=1)  # Not Here
        else:
            x_up1 = torch.cat([x_up1, x2[:,:,:x_up1.size(2)]], dim=1)  # Here

        x = self.relu(self.conv7(x_up1))
        x = self.relu(self.conv8(x))

        x_up2 = self.upconv2(x)
        if x_up1.size() == x2.size():
            x_up2 = torch.cat([x_up2, x1], dim=1)  # Not Here
        else:
            x_up2 = torch.cat([x_up2, x1[:,:,:x_up2.size(2)]], dim=1)  # Here

        x = self.relu(self.conv9(x_up2))
        x = self.relu(self.conv10(x))
        x = self.conv11(x)

        return x
