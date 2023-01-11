import asyncio, copy, os, socket, sys, time
from functools import partial
from multiprocessing import Pool, Process
from pathlib import Path
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, data, fl, log, nn, poison, resnet, sim, wandb
from cfgs.fedargs import *


# In[2]:


project = 'fl-poison-irbra'
name = 'fedavg-cnn-mnist-hdc-na-FLTrust'

#Define Custom CFGs
FLTrust["is"] = True
fedargs.agg_rule = agg.Rule.FLTrust
mal_clients = [c for c in range(0)]
#fang_attack["is"] = True
#lie_attack["is"] = True
#sota_attack["is"] = True
#label_flip_attack["is"] = True
#label_flip_attack["func"] = poison.label_flip_next
#set_lfa_labels(flip_labels = "next")

# Save Logs To File (info | debug | warning | error | critical) [optional]
log.init("info",)
wb = wandb.init(name, project)


# In[3]:


# Device settings
use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


# In[4]:


# Prepare clients
host = socket.gethostname()
clients = [host + "(" + str(client + 1) + ")" for client in range(fedargs.num_clients)]


# In[5]:


# Initialize Global and Client models
global_model = copy.deepcopy(fedargs.model)
# Load Data to clients
train_data, test_data = data.load_dataset(fedargs.dataset)


# <h2>FLTrust</h2>

# In[6]:


if FLTrust["is"]:
    train_data, FLTrust["data"] = data.random_split(train_data, FLTrust["ratio"])
    FLTrust["loader"] = torch.utils.data.DataLoader(FLTrust["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
    
    if FLTrust["proxy"]["is"]:
        FLTrust["data"], FLTrust["proxy"]["data"] = data.random_split(FLTrust["data"], FLTrust["proxy"]["ratio"])
        FLTrust["loader"] = torch.utils.data.DataLoader(FLTrust["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
        FLTrust["proxy"]["loader"] = torch.utils.data.DataLoader(FLTrust["proxy"]["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# <h2>Prepare a backdoored loader for test</h2>

# In[7]:


if backdoor_attack["is"]:
    train_data, backdoor_attack["data"] = data.random_split(train_data, backdoor_attack["ratio"])
    backdoor_attack["data"] = poison.insert_trojan(backdoor_attack["data"],
                                                   backdoor_attack["target_label"],
                                                   backdoor_attack["trojan_func"], 1)
    backdoor_attack["loader"] = torch.utils.data.DataLoader(backdoor_attack["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# <h2>Load client's data</h2>

# In[8]:


clients_data = data.split_data(train_data, clients)


# <h2>HDC DP Attack</h2>

# In[9]:


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

# In[10]:


if label_flip_attack["is"]:
    for client in mal_clients:
        clients_data[clients[client]] = label_flip_attack["func"](clients_data[clients[client]],
                                                                  label_flip_attack["labels"],
                                                                  label_flip_attack["percent"])


# <h2>Backdoor Attack</h2>

# In[11]:


if backdoor_attack["is"]:
    for client in mal_clients:
        clients_data[clients[client]] = poison.insert_trojan(clients_data[clients[client]],
                                                             backdoor_attack["target_label"],
                                                             backdoor_attack["trojan_func"], 0.5)


# In[12]:


client_train_loaders, _ = data.load_client_data(clients_data, fedargs.client_batch_size, None, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=True, **kwargs)

client_details = {
        client: {"train_loader": client_train_loaders[client],
                 "model": copy.deepcopy(global_model),
                 "model_update": None}
        for client in clients
    }


# In[13]:


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
    
    return model_update


# In[14]:


import time
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
        global_model = fl.federated_avg(client_model_updates, global_model, fedargs.agg_rule, **avgargs)
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

print(time.time() - start_time)


# <h1> End </h1>
