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
    model = ResNet(BasicBlock, args.depths, widths=tuple(args.widths))
    if use_gpu:
        model = model.cuda()
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, lossfn, args.num_epochs)
    test(model)

def train(model, optimizer, lossfn, num_epochs):
    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Finished Training')
    return model

def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    use_gpu = torch.cuda.is_available()

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
    args = CLI.parse_args()

    main()