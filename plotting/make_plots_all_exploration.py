

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    res = 50
    lw_ = 3
    max_ = 25000
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/PPO_MR-all2.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality Gap on Montezumas Revenge'
    ax3.set_title(title)

    jobs = [
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833901__2__1747602512",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833902__3__1747602512",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833529__4__1747602512",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833900__1__1747602512",
    ]

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ - $\pi$ w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ from last 1000 episodes - $\pi$ w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/PPO_MR_all2_without_RND.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = [
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833417__2__1747595815",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833418__3__1747595815",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833416__1__1747595815",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833409__4__1747595815",
    ]

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ - $\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ from last 1000 episodes - $\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps x $10^5$')
    ax3.legend()
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig(title+".svg")
    fig.savefig(title+".png")
    fig.savefig(title+".pdf")



    max_ = 25000
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/PPO_SpaceInvaders_with_E3B.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality Gap on Space Invaders'
    ax3.set_title(title)

    jobs = [
        "SpaceInvadersNoFrameskip-v4__ppo_atari__0__1747414032",
        
    ]

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ - $\pi$ w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ from last 1000 episodes - $\pi$ w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/PPO_SpaceInvaders.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = [
        "SpaceInvadersNoFrameskip-v4__ppo_atari__0__1747414032",
        
    ]

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ - $\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    label='Best $5\%$ from last 1000 episodes - $\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps x $10^5$')
    ax3.legend()
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig(title+".svg")
    fig.savefig(title+".png")
    fig.savefig(title+".pdf")