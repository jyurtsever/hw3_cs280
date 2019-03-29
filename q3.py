import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from nets import *
import torch.optim as optim
import argparse

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    print(args.depths, args.num_epochs)
    model = ResNet(BasicBlock, args.depths, widths=tuple(args.widths), num_classes=4)
    if use_gpu:
        model = model.cuda()
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, lossfn, args.num_epochs)

def train(model, optimizer, lossfn, num_epochs):
    print("Num epochs", num_epochs, "\n")
    losses = []
    for epoch in range(num_epochs):
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = data
            # print(inputs.shape, labels.shape)
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
        losses.append(running_loss/(i+1))

    print('Finished Training')
    plt.figure()
    plt.plot(list(range(num_epochs)), losses)
    plt.show()

    return model

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

    trainset = RotDataset(torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    use_gpu = torch.cuda.is_available()

    main()
