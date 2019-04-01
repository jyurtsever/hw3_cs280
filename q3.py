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
    print([2, 2, 2, 2], 4)
    print("Making model")
    model = ResNet(BasicBlock, [2, 2, 2, 2], widths=tuple((64, 128, 256, 512)), num_classes=4)
    if use_gpu:
        model = model.cuda()
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("Started training")
    model, losses, epochs = train(model, optimizer, lossfn, 15)
    
#    plt.figure()
#    plt.plot(epochs, losses)
#    plt.show()

def train(model, optimizer, lossfn, num_epochs):
    losses = []
    epochs = []
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            if i % 500 == 0:
              print(epoch, i)
            # get the inputs
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # print(inputs.shape, labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        ## Calculate training loss for current epoch
        epoch_loss = 0.
        num_batches = 0
        with torch.no_grad():
          for data in trainloader:
            inputs, labels = data
            if use_gpu:
              inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            epoch_loss += lossfn(outputs, labels).item()
            num_batches += 1
        epoch_loss /= num_batches
        losses.append(epoch_loss)
        epochs.append(epoch)

    print('Finished Training')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "saved_network.pt")
    np.save("losses.npy", np.array(losses))
    np.save("epochs.npy", np.array(epochs))
    return model, losses, epochs

if __name__ == '__main__':
#     CLI = argparse.ArgumentParser()
#     CLI.add_argument(
#         "--depths",  # name on the CLI - drop the `--` for positional/required parameters
#         nargs="*",  # 0 or more values expected => creates a list
#         type=int,
#         default=[2, 2, 2, 2],  # default if nothing is provided
#     )

#     CLI.add_argument(
#         "--widths",  # name on the CLI - drop the `--` for positional/required parameters
#         nargs="*",  # 0 or more values expected => creates a list
#         type=int,
#         default=(64, 128, 256, 512),  # default if nothing is provided
#     )

#     CLI.add_argument(
#         "--num_epochs",  # name on the CLI - drop the `--` for positional/required parameters
#         type=int,
#         default=4,  # default if nothing is provided
#     )

#     CLI.add_argument(
#         "--batch_size",  # name on the CLI - drop the `--` for positional/required parameters
#         type=int,
#         default=4,  # default if nothing is provided
#     )

    # args = {}
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    print("Started loading CIFAR")
    
    cifarset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainset = RotDataset(cifarset)
    
    print(len(cifarset))
    print(len(trainset))
    
    print("Done loading CIFAR")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    use_gpu = torch.cuda.is_available()

    main()
