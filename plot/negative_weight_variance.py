from cProfile import label
from re import S
from tkinter import N, font
from turtle import width
import matplotlib.pyplot as plt
import matplotlib.font_manager
import collections
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.ticker as ticker

""" 

docs tree requirements:
--file-group-1
  --file-name-1
    --progress.csv
  --file-name-2
    --progress.csv
--file-group-2
  --file-name-1
    --progress.csv
  --file-name-2
    --progress.csv
"""

def get_file_with_alg_groups(logdir: type=str):
    """get the file structure

    Args:
        logdir (type, str): root path. Defaults to str.

    Returns:
        _type_: dict 
        {
            'env_groups': ['env1', 'env2', 'ev3'..],
            'results_files': {
                'env1': ['alg1':{
                    ['result_file1', 'result_file2']
                    },
                        ['alg2':{
                    ['result_file3', 'result_file4']
                    }, 
                'env2': ['alg1':{
                    ['result_file1', 'result_file2']
                    },
                        ['alg2':{
                    ['result_file3', 'result_file4']
                    }, 
                }]
            }
        }   
    """
    results_dict = collections.defaultdict(str)
     
    env_groups = os.listdir(logdir)
    for env in env_groups:
        if env == '.DS_Store': continue
        alg_groups = os.listdir(f"{logdir}/{env}")
        alg_dict = collections.defaultdict(str)
        for alg in alg_groups:
            results_path = []
            if alg == '.DS_Store': continue
            for root, _, files in os.walk(f"{logdir}/{env}/{alg}"):
                for file in files:
                    if file.endswith('.csv'):
                        results_path.append(os.path.join(root, file))
                alg_dict[alg] = results_path 
        results_dict[env] = alg_dict
    return {'env_groups': env_groups, 
            'results_files': results_dict}
    

def get_file_with_single_alg(logdir: type=str):
    """get the file structure

    Args:
        logdir (type, str): root path. Defaults to str.

    Returns:
        _type_: dict 
        {
            'env1': 
                ['result_file1', 'result_file2'], 
            'env2':
                ['result_file1', 'result_file2']}
            }
        }   
    """
    return_dict = collections.defaultdict(str)
     
    env_groups = os.listdir(logdir)
    for env in env_groups:
        results = []
        if env == '.DS_Store': continue
        for root, _, files in os.walk(f"{logdir}/{env}"):
            for file in files:
                if file.endswith('.csv'):
                    results.append(os.path.join(root, file))
        return_dict[env] = results
    return return_dict

def insert(df, i, df_add):
    # 指定第i行插入一行数据
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df_new = pd.concat([df1, df_add, df2], ignore_index=True)
    return df_new

def get_results(results_file_data, key=None):
    """ return the dataframe of results

    Args:
        result_file_data (list): list of csv file paths
    Returns:
        dataframe of results ['Steps', 'normalized_return all', 'normalized_return std all']
    """
    # file_datas = pd.DataFrame(columns=['Steps', ''])
    if not key: 
        key = 'normalized_return'
    file_datas = None
    for idx, file_path in enumerate(results_file_data):
        if file_datas is None:
            file_datas = pd.read_csv(file_path)
            file_datas = file_datas[[key]]
            continue
        data = pd.read_csv(file_path)
        file_datas.insert(len(file_datas.columns), f'{key}_{idx}', data[key])
    columns = [column for column in file_datas.columns if key in file_datas.columns]
    mean = file_datas[columns].mean(axis=1)
    vars = file_datas[columns].std(axis=1)
    file_datas.insert(len(file_datas.columns), f'{key} all', mean)
    file_datas.insert(len(file_datas.columns), f'{key} std all', vars)
    file_datas = file_datas[[f'{key} all', f'{key} std all']]
    # insert steps=0's performance
    insert_data_row_0 = pd.DataFrame({f'{key} all':[0], f'{key} std all':[0]})
    file_datas = insert(file_datas, 0, insert_data_row_0)
    file_datas.insert(0, 'Steps', file_datas.index)

    return file_datas

def smooth_data(data, smooth, key):
    """
    smooth data with moving window average.
    that is,
        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    where the "smooth" param is width of that window (2k+1)
    """
    if smooth > 1:
        y = np.ones(smooth)
        # for datum in data:
        x = np.asarray(data[key])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
        data[key] = smoothed_x
    return data
    
def plots(logdir, smooth, save_path, task_name):
    """_summary_plot
    Args:
        logdir (str): path to csv dataset
        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.
    """
    sns.set(font_scale=2, )
    sns.set_style("ticks")
    palette = plt.get_cmap('Set1')

    # plt.rcParams['font.family'] = 'Times New Roman' 'calibri'
    font1 = {'family' : 'calibri',
    'weight' : 'normal',
    'size'   : 18,
    }

    results_dict = get_file_with_single_alg(logdir)
    env_groups = results_dict.keys()
    # TODO
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # "HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4" 
    # "walker-run" "cheetah-run" "fish-swim" "humanoid-stand" "humanoid-run" "quadruped-run" "hopper-hop"
    env_groups = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
    keys = ['actor_total', 'sparse_actor_total']
    colors=['#023e8a', '#1f77b4']
    axs = axs.reshape(1, -1).tolist()[0]
    my_x_ticks = np.arange(0, 1.01, 0.2)
    # my_y_ticks = np.arange(0, 110, 20)
    for idx, env in enumerate(env_groups):
        ax = axs[idx]
        for i, key in enumerate(keys):
            data = get_results(results_dict[env], key=key)
            max_step = data['Steps'].max()
            print('num of datasets', data.shape[0])
            data['Steps'] = data['Steps'] / max_step
            if smooth > 1:
                data = smooth_data(data, smooth, key)
            ax.plot(data['Steps'], 
                        data[f'{key} all'], 
                        color=colors[i], 
                        label=key, 
                        linewidth=3)
            print(f"env: {env} | key: {key}| return: {data[f'{key} all'][max_step]}| std: {data[f'{key} std all'][max_step]}")

            ax.fill_between(data['Steps'], 
                            data[f'{key} all'] - data[f'{key} std all'],
                            data[f'{key} all'] + data[f'{key} std all'], 
                            # color=colors[i], 
                            alpha=0.2)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticks(my_x_ticks)
        # ax.set_yticks(my_y_ticks)
        # ax.axis([0, 1, 0, 121])
        # ax.xaxis([0, 1])
        ax.set_xlabel('Time Steps (1e6)', fontdict=font1)
        ax.grid(True)
            
        ax.set_title(label=env, fontdict=font1, loc='center')
        if idx == 0:
            ax.legend(loc=2, fontsize=font1['size']) 
        if idx % 4 == 0:
            font2 = {'family' : 'calibri',
                'weight' : 'normal',
                'size'   : 15}
            ax.set_ylabel('Negative Weight Variance', fontdict=font2)
        

    plt.tight_layout()
    plt.savefig(save_path + task_name + '.jpg',  bbox_inches='tight')
    # plt.show()

def plots_double(logdir1, logdir2, smooth, save_path, task_name):
    """_summary_plot
    Args:
        logdir (str): path to csv dataset
        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.
    """
    sns.set(font_scale=2, )
    sns.set_style("ticks")
    palette = plt.get_cmap('Set1')

    plt.rcParams['font.family'] ='calibri'
    font1 = {'family' : 'calibri',
    'weight' : 'normal',
    'size'   : 24,
    }

    colors=['#fb5607', '#023e8a']
    labels = ['Normal', 'Heavy Tail']

    results_file_dict_1 = get_file(logdir1)
    results_file_dict_2 = get_file(logdir2)
    # plt.figure(figsize=(8.5,7.5))
    fig, axs=plt.subplots(1, 2, figsize=(10, 4))
    my_x_ticks = np.arange(0, 1.01, 0.2)
    my_y_ticks = np.arange(0, 101, 25)
    # avoid some hidden documents
    
    results_file_dicts = [results_file_dict_1, results_file_dict_2]
    for idx, results_file_dict in enumerate(results_file_dicts):
        del_dict_ = []
        if idx == 0:
            plt_ax = axs[0]
        else:
            plt_ax = axs[1]
        for file_group in results_file_dict.keys():
            if not results_file_dict[file_group]:
                del_dict_.append(file_group)
        for del_dict in del_dict_:
            del results_file_dict[del_dict]

        for idx, file_group in enumerate(results_file_dict.keys()):
            
            data = get_results(results_file_dict[file_group])
            max_step = data['Steps'].max()
            data['Steps'] = data['Steps'] / max_step
            
            plt_ax.plot(data['Steps'], 
                        data['normalized_return all'], 
                        color=colors[idx], 
                        label=labels[idx], 
                        linewidth=5)

            plt_ax.fill_between(data['Steps'], 
                            data['normalized_return all'] - data['normalized_return std all'],
                            data['normalized_return all'] + data['normalized_return std all'], 
                            color=colors[idx], 
                            alpha=0.1)
                # plt.show()
            ###设置坐标轴的粗细
            width=2
            plt_ax.spines['right'].set_visible(False)
            plt_ax.spines['top'].set_visible(False)
            plt_ax.spines['bottom'].set_linewidth(width)###设置底部坐标轴的粗细
            plt_ax.spines['left'].set_linewidth(width)####设置左边坐标轴的粗细
            plt_ax.tick_params(axis='both', which='major', labelsize=18)
            plt_ax.set_xticks(my_x_ticks)
            plt_ax.set_yticks(my_y_ticks)
            plt_ax.axis([0, 1, 0, 100])
            plt_ax.set_xlabel('Time Steps (1e6)', fontdict=font1)
            plt_ax.grid(True)
    axs[0].set_ylabel('Normalized Return', fontdict=font1)

    # plt.show()
    # ,  bbox_inches='tight'
    fig.tight_layout()
    plt.show()
    plt.savefig(f"{save_path}{task_name}.pdf", dpi=200)
     

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./negative_weight_variance')
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./plot/images/')
    parser.add_argument('--task_name', type=str, default='negative_side_variance')
    args = parser.parse_args()
    plots(args.logdir, args.smooth, args.save_path, args.task_name)
    # get_file_with_single_alg(args.logdir)