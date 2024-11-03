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
wandb.login(key="1cfab558732ccb32d573a7276a337d22b7d8b371")
import design_bench

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


def get_offline_data(nconfig):
    if nconfig.task.name != 'TFBind10-Exact-v0':
        task = design_bench.make(nconfig.task.name)
    else:
        task = design_bench.make(nconfig.task.name,
                                dataset_kwargs={"max_samples": 10000})

    offline_x = task.x
    if task.is_discrete:
        offline_x = task.to_logits(offline_x).reshape(offline_x.shape[0], -1)

    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x == 0, 1.0, std_x)
    
    offline_y = task.y
    mean_y = np.mean(offline_y, axis=0)
    std_y = np.std(offline_y, axis=0)
    
    shuffle_idx = np.random.permutation(offline_x.shape[0])

    offline_x = offline_x[shuffle_idx]
    offline_y = offline_y[shuffle_idx]
    offline_y = offline_y.reshape(-1)
    # unlabel 99% of the offline data 
    num_samples =  int(nconfig.data_ratio*offline_y.shape[0])
    offline_y[num_samples+1:]  = -1
    
    return torch.from_numpy(offline_x), torch.from_numpy(mean_x), torch.from_numpy(std_x), torch.from_numpy(offline_y), torch.from_numpy(mean_y), torch.from_numpy(std_y), num_samples


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
def tester(config, task):
    global offline_x_list, mean_x_list, std_x_list, offline_y_list, mean_y_list, std_y_list, num_samples
    offline_x = offline_x_list[config.args.seed] 
    offline_y = offline_y_list[config.args.seed]
    mean_x = mean_x_list[config.args.seed] 
    mean_y = mean_y_list[config.args.seed] 
    std_x = std_x_list[config.args.seed] 
    std_y = std_y_list[config.args.seed] 
    
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    runner.num_samples = num_samples
    runner.offline_x, runner.mean_offline_x, runner.std_offline_x = offline_x, mean_x, std_x 
    runner.offline_y, runner.mean_offline_y, runner.std_offline_y = offline_y, mean_y, std_y 
    
    return runner.test(task) 

def main():
    nconfig, dconfig = parse_args_and_config()
    wandb.init(project='BBDM-fewshot-setting',
            name=nconfig.wandb_name,
            config = dconfig) 
    args = nconfig.args
    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.device = [torch.device("cpu")]
    else:
        nconfig.training.device = [torch.device(f"cuda:{gpu_ids}")]
    
    seed_list = range(8)
    model_load_path_list = [] 
    optim_sche_load_path_list = []
    for seed in seed_list:
        nconfig.args.train = True 
        nconfig.args.seed = seed 
        model_load_path = f'results/few_shot/tune_3/{nconfig.task.name}/sampling_lr{nconfig.GP.sampling_from_GP_lr}/initial_lengthscale{nconfig.GP.initial_lengthscale}/delta{nconfig.GP.delta_lengthscale}/seed{seed}/BrownianBridge/checkpoint/top_model_epoch_100.pth'
        optim_sche_load_path = f'results/few_shot/tune_3/{nconfig.task.name}/sampling_lr{nconfig.GP.sampling_from_GP_lr}/initial_lengthscale{nconfig.GP.initial_lengthscale}/delta{nconfig.GP.delta_variance}/seed{seed}/BrownianBridge/checkpoint/top_optim_sche_epoch_100.pth'
        model_load_path_list.append(model_load_path) 
        optim_sche_load_path_list.append(optim_sche_load_path)
    
    if nconfig.task.name != 'TFBind10-Exact-v0':
        task = design_bench.make(nconfig.task.name)
    else:
        task = design_bench.make(nconfig.task.name,
                                dataset_kwargs={"max_samples": 10000})
    if task.is_discrete: 
        task.map_to_logits()
        
    global offline_x_list, mean_x_list, std_x_list, offline_y_list, mean_y_list, std_y_list, num_samples
    offline_x_list, mean_x_list, std_x_list, offline_y_list, mean_y_list, std_y_list = [],[],[],[],[],[] 
    for seed in seed_list : 
        global offline_x, mean_x, std_x, offline_y, mean_y, std_y 
        set_random_seed(seed)
        offline_x, mean_x, std_x , offline_y, mean_y , std_y, num_samples = get_offline_data(nconfig)
        offline_x = (offline_x - mean_x) / std_x
        offline_y = (offline_y - mean_y) / std_y   
        # shuffle_idx = np.random.permutation(offline_x.shape[0])
        # offline_x = offline_x[shuffle_idx]
        # offline_y = offline_y[shuffle_idx]
        offline_x = offline_x.to(nconfig.training.device[0])
        offline_y = offline_y.to(nconfig.training.device[0])
        # sorted_indices = torch.argsort(offline_y)[-128:] 
        # offline_x = offline_x[sorted_indices] 
        # offline_y = offline_y[sorted_indices] 
        
        offline_x_list.append(offline_x) 
        offline_y_list.append(offline_y) 
        mean_x_list.append(mean_x) 
        std_x_list.append(std_x) 
        mean_y_list.append(mean_y)
        std_y_list.append(std_y) 
        
    file_path = f'./few-shot-results/tuning_3_result_{nconfig.task.name}_test_{nconfig.testing.type_sampling}_l{nconfig.GP.initial_lengthscale}_d{nconfig.GP.delta_lengthscale}.csv'

    if not os.path.isfile(file_path):
        with open(file_path, 'a') as file:
            header = ['lengthscale','delta','eta','alpha','classifier_free_guidance_weight', 'mean (100th)', 'std (100th)', 'mean (80th)', 'std (80th)', 'mean (50th)', 'std (50th)']
            writer = csv.writer(file)
            writer.writerow(header)
    
    
    
    df = pd.read_csv(file_path) 
    tested_params = df[['eta','alpha','classifier_free_guidance_weight']].values.tolist()
    print(tested_params)
    lengthscale = nconfig.GP.initial_lengthscale 
    delta = nconfig.GP.delta_lengthscale 
    for eta in [0.2, 0.25, 0.5, 0.0, 0.05, 0.1, 0.15]: 
        for w in [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2 , 2.5 , 3, 4]:  
            for alpha in [0.8,0.9, 0.95, 1.0]: 
                results_100th = []
                results_80th = [] 
                results_50th = []
                
                if [eta, alpha, w] in tested_params: 
                    continue 
                # print([eta,alpha,w])
                for seed in seed_list: 
                    nconfig.model.BB.params.eta = eta 
                    nconfig.testing.classifier_free_guidance_weight = w 
                    nconfig.testing.alpha = alpha 
                    nconfig.args.train=False 
                    nconfig.args.seed = seed
                    nconfig.model.model_load_path = model_load_path_list[seed]
                    nconfig.model.optim_sche_load_path = optim_sche_load_path_list[seed]
                    result = tester(nconfig,task)
                    print("Score : ",result[0]) 
                    results_100th.append(result[0])
                    results_80th.append(result[1]) 
                    results_50th.append(result[2])
                assert len(results_100th)==8 
                np_result_100th = np.array(results_100th)
                mean_score_100th = np_result_100th.mean() 
                std_score_100th = np_result_100th.std()
                np_result_80th = np.array(results_80th)
                mean_score_80th = np_result_80th.mean() 
                std_score_80th = np_result_80th.std()
                np_result_50th = np.array(results_50th)
                mean_score_50th = np_result_50th.mean() 
                std_score_50th = np_result_50th.std()
                print(nconfig.task.name)
                print(mean_score_100th, std_score_100th)
                print(mean_score_80th, std_score_80th)
                print(mean_score_50th, std_score_50th)
                with open(file_path, 'a') as file:
                    new_row = [lengthscale,delta, eta, alpha, w, mean_score_100th, std_score_100th, mean_score_80th, std_score_80th, mean_score_50th, std_score_50th]
                    writer = csv.writer(file)
                    writer.writerow(new_row)
                    df = pd.read_csv(file_path)
                    table = wandb.Table(dataframe=df)
                    wandb.log({"data_table": table})
    
    wandb.finish() 
    
if __name__ == "__main__":
    main()
