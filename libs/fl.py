import copy
from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F
from torch import optim

import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg

def audit_attack(target, pred, flip_labels, attack_dict):
    if flip_labels is not None and len(flip_labels) > 0:
        for i in range(len(target)):
            if target[i].item() in flip_labels.keys():
                attack_dict["instances"] += 1
                if target[i] != pred[i]:
                    attack_dict["misclassifications"] += 1
                    if pred[i].item() == flip_labels[target[i].item()]:
                        attack_dict["attack_success_count"] += 1

def backdoor_test(model, backdoor_test_loader, device, source_label):
    model.eval()
    test_output = {
        "test_loss": 0,
        "accuracy": 0,
        "misclass": 0
    }
    test_loss = 0
    correct = 0
    misclass = 0

    with torch.no_grad():
        for data, target in backdoor_test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(pred)):
                if pred[i] != source_label:
                    misclass += 1
                    if pred[i] == target[i]:
                        correct += 1

    test_output["test_loss"] /= len(backdoor_test_loader.dataset)
    test_output["accuracy"] = (correct / len(backdoor_test_loader.dataset)) * 100
    test_output["misclass"] = (misclass / len(backdoor_test_loader.dataset)) * 100    

    return test_output

def client_binary(_model, data_loader, learning_rate, decay, epochs, device):
    model = copy.deepcopy(_model)
    loss = {}
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _loss = criterion(output, target)
            _loss.backward()
            optimizer.step()

        loss["Epoch " + str(epoch + 1)] = _loss.item()
    return model, loss

def client_update(_model, data_loader, learning_rate, decay, epochs, device):
    model = copy.deepcopy(_model)
    loss = {}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #_loss = F.nll_loss(output, target)
            _loss = F.cross_entropy(output, target)
            _loss.backward()
            optimizer.step()

        loss["Epoch " + str(epoch + 1)] = _loss.item()
    return model, loss
                    
def evaluate(model, test_loader, device, flip_labels = None):
    model.eval()
    test_output = {
        "test_loss": 0,
        "correct": 0,
        "accuracy": 0
    }
    
    if flip_labels is not None and len(flip_labels) > 0:
        test_output["attack"] = {
            "instances": 0,
            "misclassifications": 0,
            "attack_success_count": 0,
            "misclassification_rate": 0,
            "attack_success_rate": 0
        }

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_output["test_loss"] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            if flip_labels is not None and len(flip_labels) > 0:
                audit_attack(target, pred, flip_labels, test_output["attack"])
            test_output["correct"] += pred.eq(target.view_as(pred)).sum().item()

    test_output["test_loss"] /= len(test_loader.dataset)
    test_output["accuracy"] = (test_output["correct"] / len(test_loader.dataset)) * 100

    if flip_labels is not None and len(flip_labels) > 0:
        test_output["attack"]["attack_success_rate"] = (test_output["attack"]["attack_success_count"] /
                                                        test_output["attack"]["instances"]) * 100
        test_output["attack"]["misclassification_rate"] = (test_output["attack"]["misclassifications"] / \
                                                          test_output["attack"]["instances"]) * 100

    return test_output

def evaluate_binary(model, test_loader, device, flip_labels = None):
    criterion = torch.nn.BCELoss()
    size = 0
    model.eval()
    test_output = {
        "test_loss": 0,
        "correct": 0,
        "accuracy": 0
    }
    
    if flip_labels is not None and len(flip_labels) > 0:
        test_output["attack"] = {
            "instances": 0,
            "misclassifications": 0,
            "attack_success_count": 0,
            "misclassification_rate": 0,
            "attack_success_rate": 0
        }

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            size += len(data)
            output = model(data)
            test_output["test_loss"] += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            if flip_labels is not None and len(flip_labels) > 0:
                audit_attack(target, pred, flip_labels, test_output["attack"])
            test_output["correct"] += pred.eq(target.view_as(pred)).sum().item()

    test_output["test_loss"] /= size
    test_output["accuracy"] = (test_output["correct"] / size) * 100

    if flip_labels is not None and len(flip_labels) > 0:
        test_output["attack"]["attack_success_rate"] = (test_output["attack"]["attack_success_count"] /
                                                        test_output["attack"]["instances"]) * 100
        test_output["attack"]["misclassification_rate"] = (test_output["attack"]["misclassifications"] / \
                                                          test_output["attack"]["instances"]) * 100

    return test_output

def federated_avg(models: Dict[Any, torch.nn.Module],
                  base_model: torch.nn.Module = None,
                  rule: agg.Rule = agg.Rule.FedAvg, **kwargs) -> torch.nn.Module:
    if len(models) > 1:
        if rule is agg.Rule.FedAvg:
            model = agg.FedAvg(base_model, models)
        if rule is agg.Rule.FoolsGold:
            model = agg.FoolsGold(base_model, models, **kwargs)
        if rule is agg.Rule.FLTrust:
            model = agg.FLTrust(base_model, models, **kwargs)
        if rule is agg.Rule.FLTC:
            model = agg.FLTC(base_model, models, **kwargs)
        if rule is agg.Rule.Krum:
            model = agg.Krum(base_model, models, **kwargs)
        if rule is agg.Rule.M_Krum:
            model = agg.M_Krum(base_model, models, **kwargs)
        if rule is agg.Rule.Median:
            model = agg.Median(base_model, models, **kwargs)
        if rule is agg.Rule.T_Mean:
            model = agg.T_Mean(base_model, models, **kwargs)
        if rule is agg.Rule.DnC:
            model = agg.DnC(base_model, models, **kwargs)            
    else:
        model = copy.deepcopy(list(models.values())[0])
    return model

def train_model(_model, train_loader, lr, wd, r, device):
    model, loss = client_update(_model, train_loader, lr, wd, r, device)
    model_update = agg.sub_model(_model, model)
    return model_update, model, loss

def train_binary(_model, train_loader, lr, wd, r, device):
    model, loss = client_binary(_model, train_loader, lr, wd, r, device)
    model_update = agg.sub_model(_model, model)
    return model_update, model, loss