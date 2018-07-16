import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16) # my stone, enemy's stone, 1
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1])
        self.layer3 = self.make_layer(block, 32, layers[2])
        self.layer4 = self.make_layer(block, 32, layers[3])
        self.value_conv = nn.Conv2d(32, 1, 3, 1, 1)
        self.value_fc = nn.Linear(32 * 32, 17)
        self.value_softmax = nn.Softmax(dim=1)

        self.policy_conv1 = nn.Conv2d(32, 2, 3, 1, 1)
        self.policy_conv2 = nn.Conv2d(2, 2, 3, 1, 1)
        self.policy_softmax = nn.Softmax(dim=1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        p_out = self.policy_conv1(out)
        v_out = self. value_conv(out)

        v_out = v_out.view(out.size(0), -1)
        v_out = self.value_fc(v_out)
        v_out = self.value_softmax(v_out)

        p_out = self.policy_conv2(p_out)
        p_out = p_out.view(out.size(0), -1)
        p_out = self.policy_softmax(p_out)
        return p_out, v_out  # 1 x 2048, 1 x 17

def save_model(model, f_name):
    torch.save(model.state_dict(), f_name)

def load_model(model, f_name):
    model.load_state_dict(torch.load(f_name))



import numpy as np

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)
    np.set_printoptions(threshold=np.inf)
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

    a = np.zeros((4, 3, 32, 32))
    a = torch.Tensor(a).to(device)


    b = model(a)
    #print(b[0].shape)
    #print(torch.sum(b[0][1]))

    p_out, v_out = model(a)

    print(v_out[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    v = torch.rand(4,17).to(device)
    p = torch.rand(4, 2048).to(device)

    one = torch.sum(- v * torch.log(v_out))/4
    two = torch.sum(- p * torch.log(p_out))/4
    loss = one + two

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, v_out = model(a)

    print(v_out[0])