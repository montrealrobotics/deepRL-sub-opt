

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    res = 25
    lw_ = 3
    max_ = 1870
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    title = 'Global Optimality Gap Scaling on HalfCheetah'
    ax3.set_title(title)

    datadir = './data/PPO_HalfCheetah_4_layers.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label="4 layers"
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    datadir = './data/PPO_HalfCheetah_16_layers.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label="16 layers"
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    datadir = './data/PPO_HalfCheetah_32_layers.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label="32 layers"
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    datadir = './data/PPO_HalfCheetah_64_layers.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label="64 layers"
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    datadir = './data/PPO_HalfCheetah_128_layers.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label="128 layers"
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    datadir = './data/PPO_HalfCheetah_256_layers.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label="256 layers"
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Optimality Gap', xlabel='Steps')
    ax3.legend()
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")