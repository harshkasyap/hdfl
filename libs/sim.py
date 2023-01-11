import copy, cmath, torch
import numpy as nd
from mxnet import nd as mnd

def cosine_similarity(arr1, arr2):
    cs = mnd.dot(mnd.array(arr1), mnd.array(arr2)) / (mnd.norm(mnd.array(arr1)) + 1e-9) / (mnd.norm(mnd.array(arr2)) + 1e-9)
    return cs.asnumpy()[0]

def cosine_coord_vector_adapter(b, m, coord, dot_mb, norm_m, sim_mg, c, norm_c, norm_b, **kwargs):
    scale_norm = kwargs["scale_norm"] if "scale_norm" in kwargs else 10
    
    prev_m_coord = m[coord]
    m[coord] = cosine_coord_vector(b, m, coord, dot_mb, norm_m)
    
    _dot_mg = (sim_mg * norm_m * norm_c) - (c[coord] * (prev_m_coord - m[coord]))
    _norm_m = cmath.sqrt(norm_m**2 - prev_m_coord**2 + m[coord]**2)
    _sim_mg = (_dot_mg / (_norm_m * norm_c)).real
    
    updated = True
    if _sim_mg < sim_mg and _norm_m < (norm_b * scale_norm) and _norm_m > (norm_b * (1 / scale_norm)):
        sim_mg = _sim_mg
        norm_m = _norm_m
        dot_mb = dot_mb - b[coord] * (prev_m_coord - m[coord])
    else:
        updated = False
        m[coord] = prev_m_coord

    return m, dot_mb, norm_m, sim_mg, updated

def cosine_coord_vector(b, m, coord, dot_mb=None, norm_m = None):
    if dot_mb is None:
        dot_mb = dot(b, m)
    if norm_m is None:
        norm_m = norm(m)

    lhs = ((dot_mb / norm_m) ** 2)
    
    coeff_b_coord = b[coord]
    coeff_m_coord = m[coord]

    _dot_mb = dot_mb - (coeff_m_coord * coeff_b_coord)
    _norm_m = norm_m**2 - (coeff_m_coord**2)
    
    deg_2 = lhs - (coeff_b_coord**2)
    deg_1 = -2 * _dot_mb * coeff_b_coord
    deg_0 = lhs * _norm_m - (_dot_mb**2)
    
    d = (deg_1**2) - (4*deg_2*deg_0)
    d_sqrt = cmath.sqrt(d)
    _denom = 2*deg_2
    sol1 = ((-deg_1-d_sqrt)/(_denom)).real
    sol2 = ((-deg_1+d_sqrt)/(_denom)).real

    sol = sol1
    if abs(sol2 - coeff_m_coord) > abs(sol1 - coeff_m_coord):
        sol = sol2
    
    return sol

def dot(arr1, arr2):
    cs = mnd.dot(mnd.array(arr1), mnd.array(arr2))
    return cs.asnumpy()[0]

def eucliden_dist(arr1, arr2):
    return nd.linalg.norm(arr1-arr2)
    
def get_arr_net(_model, arr, slist):
    arr = torch.from_numpy(arr).unsqueeze(1)
    arr = arr.numpy()
    
    _param_list = []
    start_index = 0
    for shape in slist:
        end_index = start_index + nd.prod(list(shape))
        item = arr[start_index:end_index]
        start_index = end_index
        item = item.reshape(shape)
        _param_list.append(item)
    
    params = _model.state_dict().copy()
    with torch.no_grad():
        _index = 0
        for name in params:
            if "weight" in name or "bias" in name:
                params[name] = torch.from_numpy(_param_list[_index])
                _index = _index + 1

    model = copy.deepcopy(_model)
    model.load_state_dict(params, strict=False)

    return model

def get_mx_net_arr(model):
    param_list = [param.data.numpy() for param in model.parameters()]
    _param_list = nd.array(param_list).squeeze()

    arr = nd.array([[]])
    for index, item in enumerate(_param_list):
        item = item.reshape((-1, 1))
        if index == 0:
            arr = item
        else:
            arr = nd.concatenate((arr, item), axis=0)

    arr = nd.array(arr).squeeze()
    arr = mnd.array(arr)
    return arr

def get_net_arr(model):
    param_list = [param.data.numpy() for param in model.parameters()]

    arr = nd.array([[]])
    slist = []
    for index, item in enumerate(param_list):
        slist.append(item.shape)
        item = item.reshape((-1, 1))
        if index == 0:
            arr = item
        else:
            arr = nd.concatenate((arr, item), axis=0)

    arr = nd.array(arr).squeeze()
    
    return arr, slist

def grad_cosine_similarity(model1, model2):
    arr1, _ = get_net_arr(model1)
    arr2, _ = get_net_arr(model2)
    return cosine_similarity(arr1, arr2)

def grad_eucliden_dist(model1, model2):
    arr1, _ = get_net_arr(model1)
    arr2, _ = get_net_arr(model2)
    return eucliden_dist(arr1, arr2)

def grad_norm(model):
    arr, _ = get_net_arr(model)
    return norm(arr)

def grad_ssd(model1, model2):
    arr1, _ = get_net_arr(model1)
    arr2, _ = get_net_arr(model2)
    return ssd(arr1, arr2)

def max_min_norm(arr):
    if (max(arr) - min(arr)) != 0:
        return (max(arr) - arr) / (max(arr) - min(arr))
    else:
        return arr

def min_max_norm(arr):
    if (max(arr) - min(arr)) != 0:
        return (arr - min(arr)) / (max(arr) - min(arr))
    else:
        return arr

def norm(arr):
    return mnd.norm(mnd.array(arr)).asnumpy()[0]

def ssd(arr1, arr2):
    return sum((arr1-arr2)**2)

'''
import torch

def grad_norm(model, p=2):
    parameters = [param for param in model.parameters() if param.grad is not None and param.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in parameters]), p).item()

    return total_norm

def grad_cosine_similarity(model1, model2):
    cos_score=[]
    params1 = model1.state_dict().copy()
    params2 = model2.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                if len(params1[name1].shape) > 1:
                    cos_score.append(torch.nn.functional.cosine_similarity(params1[name1], params2[name1]).mean())

    return sum(cos_score)/len(cos_score)
    
def _grad_cosine_similarity(model1, model2):
    cos_score=[]
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if len(param1.shape) > 1:
            cos_score.append(torch.nn.functional.cosine_similarity(param1, param2).mean().detach().numpy())

    print(cos_score)
    return sum(cos_score)/len(cos_score)
'''