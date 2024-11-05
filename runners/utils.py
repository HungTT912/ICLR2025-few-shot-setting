import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import design_bench
# from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
# from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
# from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
# from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from gaussian_process.GP import GP 

# NAME_TO_ORACLE_DATASET = {
#     'AntMorphology-Exact-v0': AntMorphologyDataset,
#     'DKittyMorphology-Exact-v0': DKittyMorphologyDataset,
#     'TFBind8-Exact-v0': TFBind8Dataset,
#     'TFBind10-Exact-v0': TFBind10Dataset,
# }

def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    # image_path = make_dir(os.path.join(result_path, "image"))
    # log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    # sample_path = make_dir(os.path.join(result_path, "samples"))
    # sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval"))
    # print("create output path " + result_path)
    # return result_path, image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path
    return checkpoint_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))

### Sampling pretrain data from offline data 



def sampling_data_from_trajectories(x_train, y_train, num_points = 1024, threshold_diff = 0.1):
    datasets = {} 
    datasets['f0'] = {} 
    
    N, D = x_train.shape[0], x_train.shape[1] 
    points = x_train
    values = y_train  
    optima = torch.max(values)
    minima = torch.min(values)
    # optima = 1.0
    print("optima in the dataset: ", optima)

    regrets = optima - values

    num_bins = 64
    traj_len = 128
    num_trajectories = num_points 

    min_reg = torch.min(regrets)
    max_reg = torch.max(regrets)

    bin_len = (max_reg - min_reg) / num_bins
    print(min_reg, max_reg, bin_len)

    bins = [[] for i in range(num_bins)]

    for i in range(len(y_train)):
        # find the bin
        for b in range(num_bins):
            # reg = optima - data_y[i]
            if regrets[i] >= min_reg + b * bin_len and regrets[i] <= min_reg + (b + 1) * bin_len:
                bins[b].append(i)
                break

    nis = [len(i) for i in bins]
    high_exps = [-1 for i in bins]
    low_exps = [-1 for i in bins] 
    high_scores = [-1 for i in bins]
    low_scores = [-1 for i in bins]
    
    # print("90th percentile: ", np.percentile(regrets, 90))

    tau = optima - np.percentile(regrets, 90)
    K = 0.03 * N 
    print("tau: ", tau, " K: ", K)

    for b in range(len(bins)):
        low = optima - (min_reg + b * bin_len)
        high = optima - (min_reg + (b + 1) * bin_len)
        avg = (low + high) / 2
        high_exps[b] = np.exp((avg - optima) / tau)
        low_exps[b] = np.exp((avg - minima) /tau)

    for b in range(len(bins)):
        high_scores[b] = (nis[b] / (nis[b] + K)) * high_exps[b]
        low_scores[b] =  (nis[b] / (nis[b] + K)) * low_exps[b]

    scores = np.array(scores)
    scores = traj_len * (scores / np.sum(scores))
    # scores = np.floor(scores).astype(int)
    scores = np.round(scores).astype(int)
    print("Unrounded scores: ", scores, np.sum(scores))
    scores[0] += (traj_len - np.sum(scores))

    assert np.sum(scores) == traj_len
    
    selected_high_bins = np.random.choice(a=range(bin_len), size=num_points, replace=True, p=high_scores) 
    selected_low_bins = np.random.choice(a=range(bin_len), size=num_points, replace=True, p=low_scores)
    
    for i in range(num_points):
        high_index = np.random.choice(a=bins[selected_high_bins[i]], size=1) 
        low_index = np.random.choice(a=bins[selected_low_bins[i]], size=1) 
        sample = [(x_train[high_index], y_train[high_index]), (x_train[low_index],y_train[low_index])]
        datasets['f0'].append(sample) 
        
    
    return datasets 

### Sampling data from GP model
def sampling_data_from_GP(config, x_train, y_train, device, num_gradient_steps, base_gp_hyper, gp_hyper, num_samples, learning_rate, num_points, num_functions, threshold_diff = 0.1, seed=0):
    
    base_GP_Model = GP(device=base_gp_hyper.device,
                        x_train=x_train[num_samples+1:],
                        y_train=y_train[num_samples+1:],
                        lengthscale=base_gp_hyper.lengthscale,
                        variance=base_gp_hyper.variance,
                        noise = base_gp_hyper.noise,
                        mean_prior= base_gp_hyper.mean_prior)
    
    
    torch.manual_seed(seed=seed)
    datasets={}
    learning_rate_vec = torch.cat((-learning_rate*torch.ones(num_points, x_train.shape[1], device=device), learning_rate*torch.ones(num_points, x_train.shape[1], device = device)))


    for iter in range(num_functions):
        datasets[f'f{iter}']=[]
        # add noise to lengthscale and variance
        # new_lengthscale = lengthscale*(1 + delta_lengthscale*(torch.rand(1, device=device)*2 -1))
        # new_variance = variance*(1 + delta_variance*(torch.rand(1, device=device)*2 -1))
        new_base_lengthscale = base_gp_hyper.lengthscale + base_gp_hyper.delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_base_variance = base_gp_hyper.variance + base_gp_hyper.delta_variance*(torch.rand(1, device=device)*2 -1)
        base_GP_Model.set_hyper(lengthscale=new_base_lengthscale, variance=new_base_variance)
        y_train[num_samples+1:] = base_GP_Model.sampling_pseudo_label()
        best_indices = torch.argsort(y_train)[-1024:]
        best_x = x_train[best_indices]
        
        
        if config.task.name == 'TFBind8-Exact-v0': 
            selected_fit_samples = torch.randperm(x_train.shape[0])[:config.GP.num_fit_samples]
            GP_Model = GP(device=gp_hyper.device,
                        x_train=x_train[selected_fit_samples],
                        y_train=y_train[selected_fit_samples], 
                        lengthscale=gp_hyper.lengthscale, 
                        variance=gp_hyper.variance, 
                        noise=gp_hyper.noise, 
                        mean_prior=gp_hyper.mean_prior)
        else: 
            GP_Model = GP(device=gp_hyper.device,
                        x_train=x_train,
                        y_train=y_train,
                        lengthscale=gp_hyper.lengthscale, 
                        variance=gp_hyper.variance, 
                        noise=gp_hyper.noise, 
                        mean_prior=gp_hyper.mean_prior)
            
        new_lengthscale = gp_hyper.lengthscale + gp_hyper.delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_variance = gp_hyper.variance + gp_hyper.delta_variance*(torch.rand(1, device=device)*2 -1)
        # import pdb ; pdb.set_trace()
        # change lengthscale and variance of GP
        GP_Model.set_hyper(lengthscale=new_lengthscale,variance = new_variance)
        
        selected_indices = torch.randperm(best_x.shape[0])[:num_points]
        # low_y = y_pred[selected_indices].clone().detach().requires_grad_(True)
        low_x = best_x[selected_indices].clone().detach().requires_grad_(True)
        high_x = best_x[selected_indices].clone().detach().requires_grad_(True)
        joint_x = torch.cat((low_x, high_x)) 
        
        # Using gradient ascent and descent to find high and low designs 
        for t in range(num_gradient_steps): 
            mu_star = GP_Model.mean_posterior(joint_x)
            grad = torch.autograd.grad(mu_star.sum(),joint_x)[0]
            joint_x += learning_rate_vec*grad 
            # mu_star = GP_Model.mean_posterior(high_x) 
            # grad = torch.autograd.grad(mu_star.sum(),high_x)[0] 
            # high_x = high_x.add(grad, alpha = learning_rate)
        
        # high_y = GP_Model.mean_posterior(high_x)
        
        joint_y = GP_Model.mean_posterior(joint_x)
        
        low_x = joint_x[:num_points,:]
        high_x = joint_x[num_points:,:]
        low_y = joint_y[:num_points]
        high_y = joint_y[num_points:]
        
        
        for i in range(num_points):
            if high_y[i] - low_y[i] <= threshold_diff:
                continue
            sample = [(high_x[i].detach(),high_y[i].detach()),(low_x[i].detach(),low_y[i].detach())]
            datasets[f'f{iter}'].append(sample)

    
    return datasets
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        [[x_high, y_high], [x_low, y_low]] = self.data[idx]
        return (x_high, y_high), (x_low, y_low)

# Create a DataLoader for each epoch
def create_train_dataloader(data_from_GP, val_frac=0.2, batch_size=32, shuffle=True):
    train_data = []
    val_data = []
    for function, function_samples in data_from_GP.items():
        train_data = train_data + function_samples[int(len(function_samples)*val_frac):]
        val_data = val_data + function_samples[:int(len(function_samples)*val_frac)]
        
    train_dataset = CustomDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    # valid_dataset = CustomDataset(val_data)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_data

def create_val_dataloader(val_dataset, batch_size=32, shuffle=False):
    
    valid_dataset = CustomDataset(val_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return valid_dataloader

### Sampling 128 designs from offline data
def sampling_from_offline_data(x, y, n_candidates=128, type='random', percentile_sampling=0.2, seed=0):
    y = y.view(-1)
    indices = torch.argsort(y)
    x = x[indices]
    y = y[indices]
    if type == 'highest':
        print('highest')
        return x[-n_candidates:], y[-n_candidates:] 
    tmp = len(x)
    percentile_index = int(percentile_sampling * len(x))
    if type == 'low':
        x = x[:percentile_index]
        y = y[:percentile_index]
    if type == 'high':
        x = x[tmp-percentile_index:]
        y = y[tmp-percentile_index:]
    np.random.seed(seed)
    indices = np.random.choice(x.shape[0], size = n_candidates, replace=False)
    return x[indices], y[indices]
    # return x[-n_candidates:], y[-n_candidates:]
    
### Sampling 128 designs from offline data

### Testing 128 found designs by the oracle
def testing_by_oracle(task_name, high_candidates):
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        task = design_bench.make(task_name,
                                dataset_kwargs={"max_samples": 10000})
    high_candidates = high_candidates.numpy()
    score = task.predict(high_candidates)
    return score
### Testing 128 found designs by the oracle
