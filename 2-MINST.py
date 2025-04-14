# H1 = relu(XW1 +bn1)
# H2 = relu(H1W2 + b2)
# H3 = f(H2W2 + b3)

# Load data
# Build Model
# Train
# Test

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision


#  plot a curve (such as loss or accuracy) over time during training.
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'], loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


# display images from the MNIST dataset and show the corresponding label (the digit).
def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap = 'gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter(dim = 1, index = idx, value = 1)
    return out


batch_size = 512

# step1 load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download = True,
                               transform = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size = batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download = True,
                               transform = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size = batch_size, shuffle=True)


x, y = next(iter(train_loader))  #  gets a batch of data from the training set
print(x.shape, y.shape)  # torch.Size([512, 1, 28, 28]) torch.Size([512])

plot_image(x, y, 'image sample')