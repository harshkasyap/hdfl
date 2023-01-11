from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch

import copy, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg

#https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/train.py

def train_model(_model, train_loader, lr, wd, local_rounds, device):
    model = copy.deepcopy(_model)
    train_loss = {}
    optimizer = optim.SGD(model.parameters(), lr, momentum = 0.9)
    #optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    model.train()
    for epoch in range(local_rounds):
        for i_batch, data in enumerate(train_loader):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            target.sub_(1)

            inputs = Variable(inputs)
            target = Variable(target)
            logit = model(inputs)
            loss = F.nll_loss(logit, target)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 400)
            optimizer.step()
            
        train_loss["Epoch " + str(epoch + 1)] = loss.item()

    model_update = agg.sub_model(_model, model)
    return model_update, model, train_loss

def evaluate(model, data_loader, device, flip_labels):
    test_output = {
        "test_loss": 0,
        "accuracy": 0
    }
    
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, (data) in enumerate(data_loader):
        inputs, target = data
        target.sub_(1)

        size += len(target)

        inputs = Variable(inputs, volatile=True)
        target = Variable(target)
        logit = model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        accumulated_loss += F.nll_loss(logit, target, size_average = False).data
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.data.cpu().numpy().tolist()
        
    test_output["test_loss"] = (accumulated_loss / size).item()
    test_output["accuracy"] = 100.0 * (corrects / size).item()
    model.train()
    return test_output