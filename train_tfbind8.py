import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
import pandas as pd 
import csv
import wandb 

from utils import dict2namespace, get_runner, namespace2dict


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        return runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return

def trainer(config): 
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    return runner.train()
def tester(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    return runner.test() 

def main():
    nconfig, dconfig = parse_args_and_config()
    wandb.init(project='BBDM') 
    args = nconfig.args
    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.device = [torch.device("cpu")]
    else:
        nconfig.training.device = [torch.device(f"cuda:{gpu_ids}")]

    
    classifier_free_guidance_prob_list =  [0.15]
    noise_list = [1e-6,1e-4, 1e-2] 
    seed_list = range(8)
    sampling_lr_list = [0.01, 0.1, 0.05 ] 
    
    
    
    file_path = './tuning_results/tune_wandb/result/tuning_result_tfbind8.csv'
    if not os.path.isfile(file_path):
        with open(file_path, 'a') as file:
            header = ['eta','alpha','classifier_free_guidance_weight', 'mean (100th)', 'std (100th)', 'mean (80th)', 'std (80th)', 'mean (50th)', 'std (50th)']
            writer = csv.writer(file)
            writer.writerow(header)
    
    for noise in noise_list :
        nconfig.GP.noise = noise 
        for classifier_free_guidance_prob in classifier_free_guidance_prob_list : 
            for sampling_lr in sampling_lr_list: 
                nconfig.training.classifier_free_guidance_prob = classifier_free_guidance_prob
                nconfig.args.train = True  
                nconfig.model.model_load_path = None 
                nconfig.model.optim_sche_load_path = None 
                nconfig.GP.sampling_from_GP_lr = sampling_lr 

                model_load_path_list = [] 
                optim_sche_load_path_list = []
                for seed in seed_list:
                    if noise == 1e-6 and sampling_lr == 0.01 : 
                        continue 
                    if noise ==1e-6 and sampling_lr == 0.1 and seed<2: 
                        continue 
                    nconfig.args.seed = seed 
                    model_load_path, optim_sche_load_path = trainer(nconfig)
                    model_load_path_list.append(model_load_path) 
                    optim_sche_load_path_list.append(optim_sche_load_path)
                nconfig.args.train = False 
    wandb.finish() 
    
if __name__ == "__main__":
    main()
