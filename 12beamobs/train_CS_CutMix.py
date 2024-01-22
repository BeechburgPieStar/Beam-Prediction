import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from get_dataset_DA import TrainDataset_Flip, TrainDataset_Gaussian, TrainDataset_Rotate, TrainDataset_CS, TestDataset

import torch
from CNN7model import *
from torchsummary import summary
import numpy as np

class Config:
    def __init__(
        self,
        batch_size: int = 128,
        test_batch_size: int = 16,
        epochs: int = 2000,
        lr: float = 0.001,
        r: float = 1125,#250, 375， 750， 1125， 1500, 6000  #50, 150, 250, 375， 750， 1125， 1500, 6000
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.r = r

conf = Config()
writer = SummaryWriter(f"logs/CNN7_CS+CutMix_r={conf.r}")
modelweightfile = f'model/CNN7_CS+CutMix_r={conf.r}.pth'

def rand_bbox(size,lamb):
    length = size[2]
    cut_rate = 1.-lamb
    cut_length = np.int(length*cut_rate)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2

def train(model, loss, train_dataloader, optimizer, epoch, writer):
    model.train()
    correct = 0
    for data_nn in train_dataloader:
        data, target = data_nn
        target = target.long()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        lam = np.random.beta(1, 1)
        index = torch.randperm(data.size()[0]).cuda()
        
        target_a, target_b = target, target[index]

        bbx1, bbx2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1 : bbx2] = data[index, :, bbx1:bbx2]
        lam = 1 - ((bbx2-bbx1)/data.size()[-1])

        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        result_loss = lam*loss(output, target_a) + (1-lam)*loss(output, target_b)
        result_loss.backward()
        optimizer.step()
        result_loss += result_loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        result_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', result_loss, epoch)

def evaluate(model, loss, test_dataloader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    writer.add_scalar('Accuracy', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Loss/test', test_loss,epoch)

    return test_loss

def test(model, test_dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(correct / len(test_dataloader.dataset))


def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer, save_path):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch, writer)
        test_loss = evaluate(model, loss_function, val_dataloader, epoch, writer)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

def TrainDataset_prepared(r):
    x_train, x_val, y_train, y_val = TrainDataset_CS(r)
    return x_train, x_val, y_train, y_val

def TestDataset_prepared():
    x_test, y_test = TestDataset()
    return x_test, y_test


x_train, x_val, y_train, y_val = TrainDataset_prepared(conf.r)

train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

model = base_model()
optim = torch.optim.Adam(model.parameters(), lr=conf.lr)
if torch.cuda.is_available():
    model = model.cuda()
    summary(model, (2, 128))

loss = nn.NLLLoss()
if torch.cuda.is_available():
    loss = loss.cuda()

train_and_evaluate(model, 
    loss_function=loss, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    optimizer=optim, 
    epochs=conf.epochs, 
    writer=writer, 
    save_path=modelweightfile)

x_test, y_test = TestDataset_prepared()
test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
model = torch.load(modelweightfile)
test(model,test_dataloader)