import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import glob

def smooth(y, sm):
    return gaussian_filter(y, sm, mode='nearest')

def plot(attr, dfs, x_axis=None, sm=4,maxNum=-1, figsize=(8,4), ax=None,**kwargs):
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    for name,df in dfs.items():
        smoothed = smooth(df[attr][:maxNum],sm)
        if x_axis:
            ax.plot(df[x_axis][:maxNum], smoothed, label=name,**kwargs)
        else:
            ax.plot(smoothed, label=name,**kwargs)
    ax.set_title(attr)
    ax.legend()

def plot_with_error(name, all_dfs, x_axis=None, max_size=-1, legend=False,figsize=(8,4), sm=0, ax=None, quantile=False, traces=False, traces_only=False, order=None,**kwargs):
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    
    if order is None:
        order = all_dfs.keys()
        
    for n, k in enumerate(order):
        dfs = all_dfs[k]
        df_concat = pd.concat(dfs, sort=False)
        by_row_index = df_concat.groupby(df_concat.index)
        
        mean_df = by_row_index.mean()
        std_df = by_row_index.std()
        lower_df = mean_df - std_df
        upper_df = mean_df + std_df
        median_df = mean_df
        
        if quantile:
            median_df = by_row_index.median()
            lower_df = by_row_index.quantile(0.25)
            upper_df = by_row_index.quantile(0.75)

        if x_axis:
            new_max_size = 2 if traces_only else max_size
            ax.plot(median_df[x_axis][:new_max_size], smooth(median_df[name],sm)[:new_max_size], label=k, **kwargs)
            ax.fill_between(
                median_df[x_axis][:new_max_size],
                smooth(lower_df[name],sm)[:new_max_size],
                smooth(upper_df[name],sm)[:new_max_size],
                alpha=.3,
            )

        if traces or traces_only:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color = colors[n % len(colors)]
            for df in dfs:
                ax.plot(df[x_axis][:max_size], smooth(df[name],sm)[:max_size],
                        c=color, alpha=(0.7 if traces_only else 0.2))
                

        else:
            if not traces_only:
                ax.plot(smooth(median_df[name],sm)[:max_size], label=k, **kwargs)
                ax.fill_between(
                    range(len(median_df[name][:max_size])),
                    smooth(lower_df[name],sm)[:max_size],
                    smooth(upper_df[name],sm)[:max_size],
                    alpha=.3,
                )
            if traces or traces_only:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                color = colors[n % len(colors)]
                for df in dfs:
                    ax.plot(smooth(df[name],sm)[:max_size],
                            c=color, alpha=(0.7 if traces_only else 0.2))
    ax.set_title(name)
    if legend:
        ax.legend()


def get_dfs_possible(names, print_error=False):
    dfs = []
    for name in names:
        try:
            df = pd.read_csv(name)
            dfs.append(df)
        except:
            if print_error: print("Couldn't load %s"%name)
    return dfs

def load_all_dfs(folder_name):
    return get_dfs_possible(glob.glob('%s/**/progress.csv'%folder_name,recursive=True))


def load_experiment(folder_name):
    folders = {k.split('/')[-2]: k for k in glob.glob("%s/*/"%folder_name)}
    return {
        k: get_dfs_possible(glob.glob('%s/**/progress.csv'%v,recursive=True))
        for k, v in folders.items()
    }

def aggregate_mean(dfs):
    df_concat = pd.concat(dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    return by_row_index.mean()

def aggregate_std(dfs):
    df_concat = pd.concat(dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    return by_row_index.std()