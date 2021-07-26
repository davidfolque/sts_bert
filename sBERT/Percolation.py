import numpy as np
from random import randint
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # this includes tensor functions that we can use in backwards pass
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torch.optim as optim


# this function does a depth-first search across a matrix of 0s and 1s,
# looking for a path of 1s between the first and last columns of the matrix
# such a path is called a 'percolation path'

def percolate(mat):
    """
    Returns True if there is a percolation path of 1s from col 0 col -1 of a matrix of 0s and 1s
    """
    nrows = mat.shape[0]
    ncols = mat.shape[1]
    frontier = set()
    for i in range(0, nrows):
        if mat[i, 0]:
            frontier.add((i, 0))
    explored = set()
    flag = False  # this will be returned if the frontier becomes empty without finding a
    # filled pixel in the right-most column
    while frontier:  # frontier evaluates to True in this context if it is non-empty
        r, c = frontier.pop()
        explored.add((r, c))
        if r > 0:  # North
            if mat[r - 1, c]:
                coords = (r - 1, c)
                if coords not in explored:
                    if coords not in frontier:  # this order of testing is necessary since each element of explored has been in frontier
                        frontier.add(coords)
        if c < ncols - 1:  # East
            if mat[r, c + 1]:
                if c + 1 == mat.shape[1] - 1:  #  Hurray, we have percolated to the last column
                    flag = True
                    break
                coords = (r, c + 1)
                if coords not in explored:
                    if coords not in frontier:  # this order of testing is necessary since each element of explored has been in frontier
                        frontier.add(coords)
        if r < nrows - 1:  # South
            if mat[r + 1, c]:
                coords = (r + 1, c)
                if coords not in explored:
                    if coords not in frontier:  # this order of testing is necessary since each element of explored has been in frontier
                        frontier.add(coords)
        if c > 0:  # West
            if mat[r, c - 1]:
                coords = (r, c - 1)
                if coords not in explored:
                    if coords not in frontier:  # this order of testing is necessary since each element of explored has been in frontier
                        frontier.add(coords)
    return flag



def make_percolation_dataset(side=8, threshold=0.42, n_examples=10):
    """
    This function generates an array of random images, in the form needed for Keras, and
    then labels them as percolati# ng or not, using the percolate function. Roughly 50% of the
    images will have class 1 (percolating), so the dataset is likely to be reasonably balanced.
    """
    X_data = (np.random.random([n_examples, side, side, 1]) > threshold).astype(float)
    Y_data = np.zeros([n_examples, 1])
    for i in range(0, n_examples):
        if percolate(X_data[i, :, :, 0]):
            Y_data[i, 0] = 1
    dataset = [{'image': torch.tensor(x.astype(np.float32)).reshape(1, side, side),
                'label': torch.tensor(float(y)).reshape(1)} for x, y in zip(X_data, Y_data)]
    print(sum([x['label'].item() for x in dataset]) / len(dataset))
    return dataset


# this function calculates the loss and error rate on the validation set
# assuming that the output of the neural net nn is a single prediction probability, from a single sigmoid neuron.

def accuracy_and_loss(net, dataloader):
    loss_function = nn.BCELoss()
    total_correct = 0
    total_loss = 0.0
    total_examples = 0
    n_batches = 0
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            batch_loss = loss_function(outputs, labels)  # this is averaged over the batch
            n_batches += 1
            total_loss += batch_loss.item()
            total_correct += sum(
                (outputs > 0.5) == (labels > 0.5)).item()  # number correct in the minibatch
            total_examples += labels.size(
                0)  # the number of labels, which is just the size of the minibatch

    accuracy = total_correct / total_examples
    mean_loss = total_loss / n_batches

    return (accuracy, mean_loss)  # print( "Accuracy on test set: %d %%" %(100 * correct/total_examples))


# a nn with one hidden layer
class NN_one_hidden(nn.Module):

    def __init__(self):
        super(NN_one_hidden, self).__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(64, 100),  # try 100 hidden neurons
            nn.ReLU(),
            # note that we need the brackets after nn.ReLU() and nn.Sigmoid(), because these
            # are classes that need to be instantiated
            nn.Linear(100, 1),
            nn.Sigmoid())  # we are predicting only two classes, so we can use one sigmoid neuron as output

    def forward(self, x):  # computes the forward pass ... this one is particularly simple
        x = self.layers(x)
        return x

    def __getattr__(self, item):
        if item == 'device':
            try:
                return torch.cuda.get_device_name(self.layers[1].weight.data.get_device())
            except RuntimeError:
                return 'cpu'
        return super().__getattr__(item)


class NN_two_hidden(nn.Module):

    def __init__(self):
        super(NN_two_hidden, self).__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(64, 100),  # try 100 hidden neurons
            nn.ReLU(),
            # note that we need the brackets after nn.ReLU() and nn.Sigmoid(), because these
            # are classes that need to be instantiated
            nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1),
            nn.Sigmoid())  # we are predicting only two classes, so we can use one sigmoid neuron as output

    def forward(self, x):  # computes the forward pass ... this one is particularly simple
        x = self.layers(x)
        return x

    def __getattr__(self, item):
        if item == 'device':
            try:
                return torch.cuda.get_device_name(self.layers[1].weight.data.get_device())
            except RuntimeError:
                return 'cpu'
        return super().__getattr__(item)


def define_and_train(NN_class, args, n_epochs, training_set, test_set, batch_size=32,
                     weight_decay=0.0, continue_net=None, pr=False):
    loss_function = nn.BCELoss()
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    if continue_net == None:
        thenet = NN_class(**args)
    else:
        thenet = continue_net
    optimizer1 = optim.Adam(thenet.parameters(), weight_decay=weight_decay, lr=0.001)
    # optimizer1 = optim.SGD( thenet.parameters(), weight_decay=weight_decay , lr = 0.01,
    # momentum = 0.9)

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    conv3x3_weight_grads = []
    conv1x1_weight_grads = []
    bias_grads = []

    start_time = time()

    for epoch in range(n_epochs):  # number of times to loop over the dataset

        total_loss = 0
        total_correct = 0
        total_examples = 0
        n_mini_batches = 0
        conv3x3_weight_grad = torch.zeros(3, 3)

        errors = []

        thenet.train()
        for i, mini_batch in enumerate(trainloader, 0):
            images, labels = mini_batch

            # zero the parameter gradients
            # all the parameters that are being updated are in the optimizer,
            # so if we zero the gradients of all the tensors in the optimizer,
            # that is the safest way to zero all the gradients
            optimizer1.zero_grad()

            outputs = thenet(images)  # this is the forward pass

            loss = loss_function(outputs, labels)

            loss.backward()  # does the backward pass and computes all gradients

            # conv3x3_weight_grad += thenet.conv3x3.weight.grad.data[0,0]

            optimizer1.step()  # does one optimisation step

            n_mini_batches += 1  # keep track of number of minibatches, and collect the loss for each minibatch
            total_loss += loss.item()  # remember that the loss is a zero-order tensor
            # so that to extract its value, we use .item(), as we cannot index as there are no dimensions

            # keep track of number of examples, and collect number correct in each minibatch
            total_correct += sum((outputs > 0.5) == (labels > 0.5)).item()
            total_examples += len(labels)

            for j in range(len(labels)):
                if len(errors) < 10 and (outputs[j] > 0.5) != (labels[j] > 0.5):
                    errors.append(images[j])

        # calculate statistics for each epoch and print them.
        # You can alter this code to accumulate these statistics into lists/vectors and plot them
        epoch_training_accuracy = total_correct / total_examples
        epoch_training_loss = total_loss / n_mini_batches
        # conv3x3_weight_grad /= n_mini_batches

        thenet.eval()
        epoch_val_accuracy, epoch_val_loss = accuracy_and_loss(thenet, testloader)
        duration = int(time() - start_time)

        print('Epoch %d loss: %.3f acc: %.3f val_loss: %.3f val_acc: %.3f time: %d:%d%d' % (
        epoch + 1, epoch_training_loss, epoch_training_accuracy, epoch_val_loss, epoch_val_accuracy,
        duration // 60, duration % 60 // 10, duration % 60 % 10))

        # if (epoch + 1) % 10 == 0:
        # print(thenet.conv3x3.weight)
        # print(thenet.conv3x3.bias)
        # print(thenet.conv1x1.weight)

        if pr:
            plt.imshow(errors[0].numpy().squeeze())
            plt.show()

        train_loss.append(epoch_training_loss)
        train_acc.append(epoch_training_accuracy)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_accuracy)
        conv3x3_weight_grads.append(conv3x3_weight_grad)

    history = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss,
               'val_acc': val_acc, 'conv3x3_weight_grads': conv3x3_weight_grads,
               'conv1x1_weight_grads': conv1x1_weight_grads, 'bias_grads': bias_grads}
    return (history, thenet)

