#!/usr/bin/env python
# coding: utf-8

# In[47]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import asyncio, copy, os, socket, sys, time
from functools import partial
from multiprocessing import Pool, Process
from pathlib import Path
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from libs import agg, data, fl, log, nn, poison, resnet, sim, wandb
from cfgs.fedargs import *


# In[57]:


project = 'fl-poison-hdfl'
name = 'Mnist-HDC-NA-FedAvg'

#Define Custom CFGs
#FLTrust["is"] = True
#fedargs.agg_rule = agg.Rule.FedAvg
#mal_clients = [c for c in range(8)]
#fang_attack["is"] = True
#lie_attack["is"] = True
#sota_attack["is"] = True
#label_flip_attack["is"] = True
#label_flip_attack["func"] = poison.label_flip_next
#set_lfa_labels(flip_labels = "next")

# Save Logs To File (info | debug | warning | error | critical) [optional]
log.init("info",)
#wb = wandb.init(name, project)


# In[58]:


import numpy as nd

'''
r_proj = nd.random.randint(2, size=(10000,10000))
r_proj[r_proj == 0] = -1
r_inv_proj = nd.linalg.pinv(r_proj)

print(r_proj.shape, r_inv_proj.shape)

with open('proj.npy', 'wb') as f:
    nd.save(f, r_proj)
    
with open('inv.npy', 'wb') as f:
    nd.save(f, r_inv_proj)  
'''

with open('proj.npy', 'rb') as f:
    r_proj = nd.load(f)

with open('inv.npy', 'rb') as f:
    r_inv_proj = nd.load(f)    


# In[59]:


import sklearn.metrics.pairwise as smp

def DnC(model_list, **kwargs):
    num_buckets=1
    bucket=100000
        
    all_updates = torch.tensor(model_list)
    n, d = all_updates.shape

    n_attackers = kwargs["beta"]

    final_indices = []
    
    for p in nd.arange(num_buckets):
        idx = nd.sort(nd.random.choice(d, bucket, replace=False))
        sampled_all_updates = all_updates[:, idx]
        sampled_good_updates = all_updates[n_attackers:][:, idx]

        centered_all_updates = sampled_all_updates - torch.mean(sampled_all_updates, 0)
        centered_good_updates = sampled_good_updates - torch.mean(sampled_good_updates, 0)
        
        u, s, v = torch.svd(centered_all_updates)
        u_g, s_g, v_g = torch.svd(centered_good_updates)
        
        scores = torch.mm(centered_all_updates, v[:,0][:, None]).cpu().numpy()
        
        final_indices.append(list(nd.argsort(scores[:,0]**2)[:(n-int(1.5*n_attackers))]))

    result = set(final_indices[0]) 
    for currSet in final_indices[1:]: 
        result.intersection_update(currSet)
    final_idx = nd.array(list(result))

    model = nd.array(all_updates[final_idx]).mean(axis=0)
    return model

def FoolsGold(model_list, **kwargs):
    len_grad = len(model_list[0])
    n_clients = len(model_list)

    cs = smp.cosine_similarity(model_list) - nd.eye(n_clients)
    maxcs = nd.max(cs, axis=1)
    
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (nd.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / nd.max(wv)
    wv[(wv == 1)] = .99
    
    # Logit function
    wv = (nd.log(wv / (1 - wv)) + 0.5)
    wv[(nd.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    updated_model_list = []
    for index, model in enumerate(model_list):
        model = model * wv[index]
        updated_model_list.append(model)
                                  
    model = nd.array(updated_model_list).mean(axis=0)
    return model

def FL_Trust(model_list, **kwargs):
    base_model_update = get_enc_model(kwargs["base_model_update"])
    base_norm = kwargs["base_norm"] if "base_norm" in kwargs else True

    if base_norm:
        # Base Model Norm
        base_model_update_norm = sim.norm(base_model_update)

    ts_score_list=[]
    fl_score_list=[]
    updated_model_list = []
    for model in model_list:
        ts_score = sim.cosine_similarity(base_model_update, model)

        # Relu
        if ts_score < 0:
            ts_score = 0
        ts_score_list.append(ts_score)

        if base_norm:
            # Model Norm    
            norm = sim.norm(model)
            ndiv = base_model_update_norm/norm
            scale_norm = ts_score * ndiv
            model = model * scale_norm
            fl_score_list.append(scale_norm)
        else:
            model = model * ts_score

        updated_model_list.append(model)
    
    log.info("Cosine Score {}".format(ts_score_list))
    log.info("FLTrust Score {}".format(fl_score_list))
        
    model = nd.array(updated_model_list).sum(axis=0) / sum(ts_score_list)

    return model

def M_Krum(model_list, **kwargs):
    beta = kwargs["beta"]
    lb = beta//2
    ub = len(model_list) - beta//2 - 1

    euclidean_dists = []
    for index1, model1 in enumerate(model_list):
        model_dists = []
        for index2, model2 in enumerate(model_list):
            if index1 != index2:
                dist = sim.eucliden_dist(model1, model2)
                model_dists.append(dist)
        sq_dists = torch.sum(torch.sort(torch.tensor(model_dists)).values[lb:ub])
        euclidean_dists.append(sq_dists)
            
    min_model_indices = nd.argpartition(nd.array(euclidean_dists), len(model_list) - 2*beta - 2)
    min_model_indices = min_model_indices[:len(model_list) - 2*beta - 2]
    log.info("M_Krum Candidates are {}".format([index for index in min_model_indices]))
    
    model_list = [model for index, model in enumerate(model_list) if index in min_model_indices]
    model = nd.array(model_list).mean(axis=0)
    
    return model

def M_Cos(model_list, **kwargs):
    beta = kwargs["beta"]
    lb = beta//2
    ub = len(model_list) - beta//2 - 1

    cosine_dists = []
    for index1, model1 in enumerate(model_list):
        model_dists = []
        for index2, model2 in enumerate(model_list):
            if index1 != index2:
                dist = 1 - sim.cosine_similarity(model1, model2)
                model_dists.append(dist)
        cos_sims = torch.sum(torch.sort(torch.tensor(model_dists)).values[lb:ub])
        cosine_dists.append(cos_sims)
            
    min_model_indices = nd.argpartition(nd.array(cosine_dists), len(model_list) - 2*beta - 2)
    min_model_indices = min_model_indices[:len(model_list) - 2*beta - 2]
    log.info("M_Cos Candidates are {}".format([index for index in min_model_indices]))
    
    model_list = [model for index, model in enumerate(model_list) if index in min_model_indices]
    model = nd.array(model_list).mean(axis=0)
    
    return model

def T_Mean(model_list, **kwargs):
    beta = kwargs["beta"]
    lb = beta
    ub = len(model_list) - beta

    updated_model_list = [model for model in model_list]
    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.sort(torch.stack([model for model in updated_model_tensors], 0), dim = 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors.values, 0, 1).numpy()
    
    model = nd.zeros(len(model_list[0]))
    for index, arr in enumerate(merged_updated_model_arrs):
        model[index] = arr[lb:ub].mean(0)

    return model


# In[60]:


def get_enc_model(model):
    arr, slist = sim.get_net_arr(model)

    rem = nd.zeros(10000- (len(arr) % 10000))
    if len(arr) % 10000 != 0:
        arr = nd.concatenate((arr, rem), axis=None)

    #enc_model = []
    enc_model = nd.array([])
    index = 0
    while index < len(arr):
        #enc_model.append(arr[index:index+10000] @ r_proj)
        enc_model = nd.concatenate((enc_model, (arr[index:index+10000] @ r_proj)), axis = None)
        index = index + 10000
        #print(index)

    return enc_model

def get_enc_agg(models, **kwargs):
    model_list = list(models.values())
    enc_model = nd.array(model_list).mean(axis=0)
    if fedargs.agg_rule == agg.Rule.DnC:
        enc_model = DnC(model_list, **kwargs)
    if fedargs.agg_rule == agg.Rule.M_Krum:
        enc_model = M_Krum(model_list, **kwargs)
    if fedargs.agg_rule == agg.Rule.M_Cos:
        enc_model = M_Cos(model_list, **kwargs)
    if fedargs.agg_rule == agg.Rule.T_Mean:
        enc_model = T_Mean(model_list, **kwargs)
    if fedargs.agg_rule == agg.Rule.FLTrust:
        enc_model = FL_Trust(model_list, **kwargs)
    if fedargs.agg_rule == agg.Rule.FoolsGold:
        enc_model = FoolsGold(model_list, **kwargs)
    return enc_model

def get_dec_model(enc_model):
    arr, slist = sim.get_net_arr(fedargs.model)
    
    rem = nd.zeros(10000- (len(arr) % 10000))
    if len(arr) % 10000 != 0:
        arr = nd.concatenate((arr, rem), axis=None)
    
    dec_model = nd.zeros(len(arr))
    index = 0
    while index < len(arr):
        dec = enc_model[index: index + 10000] @ r_inv_proj
        dec_model[index: index + 10000] = dec
        index = index + 10000
        #print(index)

    dec_model = sim.get_arr_net(fedargs.model, dec_model, slist)
    return dec_model


# In[61]:


# Device settings
use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


# In[62]:


# Prepare clients
host = socket.gethostname()
clients = [host + "(" + str(client + 1) + ")" for client in range(fedargs.num_clients)]


# In[63]:


# Initialize Global and Client models
global_model = copy.deepcopy(fedargs.model)
# Load Data to clients
train_data, test_data = data.load_dataset(fedargs.dataset)


# <h2>FLTrust</h2>

# In[64]:


if FLTrust["is"]:
    train_data, FLTrust["data"] = data.random_split(train_data, FLTrust["ratio"])
    FLTrust["loader"] = torch.utils.data.DataLoader(FLTrust["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
    
    if FLTrust["proxy"]["is"]:
        FLTrust["data"], FLTrust["proxy"]["data"] = data.random_split(FLTrust["data"], FLTrust["proxy"]["ratio"])
        FLTrust["loader"] = torch.utils.data.DataLoader(FLTrust["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
        FLTrust["proxy"]["loader"] = torch.utils.data.DataLoader(FLTrust["proxy"]["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# <h2>Prepare a backdoored loader for test</h2>

# In[65]:


if backdoor_attack["is"]:
    train_data, backdoor_attack["data"] = data.random_split(train_data, backdoor_attack["ratio"])
    backdoor_attack["data"] = poison.insert_trojan(backdoor_attack["data"],
                                                   backdoor_attack["target_label"],
                                                   backdoor_attack["trojan_func"], 1)
    backdoor_attack["loader"] = torch.utils.data.DataLoader(backdoor_attack["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# <h2>Load client's data</h2>

# In[67]:


iidargs = {"non_iid": False, "rate_unbalance": 0.25}
clients_data = data.split_data(train_data, clients, **iidargs)


# <h2>HDC DP Attack</h2>

# In[68]:


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def hdc_train(hdc_data, device, hdc_args):
    hdc_data_loader = torch.utils.data.DataLoader(hdc_data, batch_size=len(hdc_data), shuffle=True)
    hdc_model = hdc.HDC(hdc_args["one_d_len"], hdc_args["hdc_proj_len"], len(hdc_args["labels"]), device)
    train_acc = hdc_model.train(hdc_data_loader, device)
    return hdc_model

if hdc_dp_attack["is"]:
    hdc_tasks = [hdc_train(clients_data[clients[client]], device,
                            hdc_dp_attack["args"]) for client in mal_clients]
    try:
        hdc_models = fedargs.loop.run_until_complete(asyncio.gather(*hdc_tasks))
    except KeyboardInterrupt as e:
        log.error("Caught keyboard interrupt. Canceling hdc_dps...")
        hdc_tasks.cancel()
        fedargs.loop.run_forever()
        hdc_tasks.exception()

    hdc_clients_data = {client: (clients_data[clients[client]], hdc_models[index])
                        for index, client in enumerate(mal_clients)}

    mal_clients_data = hdc_dp_attack["func"](hdc_clients_data,
                                             hdc_dp_attack["args"],
                                             label_flip_attack["labels"],
                                             hdc_dp_attack["percent"])

    for client, mal_data in enumerate(mal_clients_data):
        clients_data[clients[client]] = mal_data


# <h2>Label Flip Attack</h2>

# In[69]:


if label_flip_attack["is"]:
    for client in mal_clients:
        clients_data[clients[client]] = label_flip_attack["func"](clients_data[clients[client]],
                                                                  label_flip_attack["labels"],
                                                                  label_flip_attack["percent"])


# <h2>Backdoor Attack</h2>

# In[70]:


if backdoor_attack["is"]:
    for client in mal_clients:
        clients_data[clients[client]] = poison.insert_trojan(clients_data[clients[client]],
                                                             backdoor_attack["target_label"],
                                                             backdoor_attack["trojan_func"], 0.5)


# In[71]:


client_train_loaders, _ = data.load_client_data(clients_data, fedargs.client_batch_size, None, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=True, **kwargs)

client_details = {
        client: {"train_loader": client_train_loaders[client],
                 "model": copy.deepcopy(global_model),
                 "model_update": None}
        for client in clients
    }


# In[72]:


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def process(client, epoch, model, train_loader, fedargs, device):
    # Train
    model_update, model, loss = fedargs.train_func(model, train_loader, 
                                                   fedargs.learning_rate,
                                                   fedargs.weight_decay,
                                                   fedargs.local_rounds, device)

    log.jsondebug(loss, "Epoch {} of {} : Federated Training loss, Client {}".format(epoch, fedargs.epochs, client))
    log.modeldebug(model_update, "Epoch {} of {} : Client {} Update".format(epoch, fedargs.epochs, client))
    
    enc_model_update = get_enc_model(model_update)
    return model_update


# In[ ]:


##### import time
start_time = time.time()
    
# Federated Training
for epoch in tqdm(range(fedargs.epochs)):
    log.info("Federated Training Epoch {} of {}".format(epoch, fedargs.epochs))

    # Global Model Update
    if epoch > 0:
        # For Tmean and FLTrust, not impacts others as of now
        avgargs = {"beta": len(mal_clients), 
                   "base_model_update": global_model_update if FLTrust["is"] else None,
                   "base_norm": True}
        
        # Average
        #global_model = fl.federated_avg(client_model_updates, global_model, fedargs.agg_rule, **avgargs)
        model_update = get_dec_model(get_enc_agg(client_model_updates, **avgargs))
        global_model = agg.sub_model(global_model, model_update)
        log.modeldebug(global_model, "Epoch {}: Server Update".format(epoch))
        
        # Test, Plot and Log
        global_test_output = fedargs.eval_func(global_model, test_loader, device, label_flip_attack["labels"])
        wb.log({"epoch": epoch, "time": time.time(), "acc": global_test_output["accuracy"], "loss": global_test_output["test_loss"]})
        log.jsoninfo(global_test_output, "Global Test Outut after Epoch {} of {}".format(epoch, fedargs.epochs))
        
        # Evaluate LFA
        if "attack" in global_test_output:
            if "attack_success_rate" in global_test_output["attack"]:
                wb.log({"attack_success_rate": global_test_output["attack"]["attack_success_rate"]})
            if "misclassification_rate" in global_test_output["attack"]:
                wb.log({"misclassification_rate": global_test_output["attack"]["misclassification_rate"]})

        # Evaluate Backdoor
        if backdoor_attack["is"]:
            backdoor_test_output = fl.backdoor_test(global_model, backdoor_attack["loader"], device, backdoor_attack["target_label"])
            wb.log({"backdoor_success_rate": backdoor_test_output["accuracy"]})
            log.jsoninfo(backdoor_test_output, "Backdoor Test Outut after Epoch {} of {}".format(epoch, fedargs.epochs))

        # Update client models
        for client in clients:
            client_details[client]['model'] = copy.deepcopy(global_model)

    # Clients
    tasks = [process(client, epoch, client_details[client]['model'],
                     client_details[client]['train_loader'],
                     fedargs, device) for client in clients]
    try:
        updates = fedargs.loop.run_until_complete(asyncio.gather(*tasks))
    except KeyboardInterrupt as e:
        log.error("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        fedargs.loop.run_forever()
        tasks.exception()

    for client, update in zip(clients, updates):
        client_details[client]['model_update'] = update
    client_model_updates = {client: details["model_update"] for client, details in client_details.items()}
    
    # Fang attack
    if fang_attack["is"]:
        client_model_updates = fang_attack["func"](client_model_updates, len(mal_clients), fang_attack["kn"])
        
    # LIE attack
    if lie_attack["is"]:
        client_model_updates = lie_attack["func"](client_model_updates, len(mal_clients), lie_attack["kn"])
   
    # SOTA attack
    if sota_attack["is"]:
        client_model_updates = sota_attack["func"](client_model_updates, len(mal_clients), 
                                                   sota_attack["kn"], sota_attack["dev_type"])
    
    # FLtrust or FLTC based aggregation rules or attacks
    if FLTrust["is"]:
        global_model_update, _, _ = fedargs.train_func(global_model, FLTrust["loader"],
                                                       fedargs.learning_rate,
                                                       fedargs.weight_decay,
                                                       fedargs.local_rounds, device)

        # For Attacks related to FLTrust
        base_model_update = global_model_update
        if FLTrust["proxy"]["is"]:
            base_model_update, _, _ = fedargs.train_func(global_model, FLTrust["proxy"]["loader"],
                                                         fedargs.learning_rate,
                                                         fedargs.weight_decay,
                                                         fedargs.local_rounds, device)
        
        # Layer replacement attack
        if layer_replacement_attack["is"]:
            for client in mal_clients:
                client_details[clients[client]]['model_update'] = layer_replacement_attack["func"](base_model_update,
                                                                                                   client_details[clients[client]]['model_update'],
                                                                                                   layer_replacement_attack["layers"])

        # For cosine attack, Malicious Clients
        if cosine_attack["is"]:
            p_models, params_changed = cosine_attack["func"](base_model_update, cosine_attack["args"], epoch,
                                                             client_model_updates, len(mal_clients), cosine_attack["kn"])
            
            for client, p_model in enumerate(p_models):
                client_details[clients[client]]['model_update'] = p_model 

            #plot params changed for only one client
            fedargs.tb.add_scalar("Params Changed for Cosine Attack/", params_changed, epoch)

        # For sybil attack, Malicious Clients
        if sybil_attack["is"]:
            for client in mal_clients:
                client_details[clients[client]]['model_update'] = base_model_update
                
        # again pair, as changed during attack
        client_model_updates = {client: details["model_update"] for client, details in client_details.items()}
    
    client_model_updates = {client: get_enc_model(details["model_update"]) for client, details in client_details.items()}

print(time.time() - start_time)


# <h1> End </h1>
