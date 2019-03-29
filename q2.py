import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from oldNets import *
import torch.optim as optim
import argparse

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    print(args.depths, args.num_epochs)
    model = ResNet(BasicBlock, args.depths) #widths=tuple(args.widths))
    if use_gpu:
        model = model.cuda()
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, lossfn, args.num_epochs)
    test(model)

def train(model, optimizer, lossfn, num_epochs):
    losses = []
    iterations = []
    epoch = 1
    #
    return model, losses, iterations

def test(model):
    correct_test = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct_test / total))
    correct_train = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    print('Accuracy of the network on the 50000 train images: %d %%' % (
            100 * correct_train / total))


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--depths",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[2, 2, 2, 2],  # default if nothing is provided
    )

    CLI.add_argument(
        "--widths",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=(64, 128, 256, 512),  # default if nothing is provided
    )

    CLI.add_argument(
        "--num_epochs",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=4,  # default if nothing is provided
    )

    CLI.add_argument(
        "--batch_size",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=4,  # default if nothing is provided
    )

    args = CLI.parse_args()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False,)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    use_gpu = torch.cuda.is_available()



    main()
