
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *


if __name__ == '__main__':

    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Local Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/MinAtar_SpaceInvaders_local_optimality_gap.csv'
    title = 'Local Optimality for SpaceInvaders'
    df = pd.read_csv(datadir)
    ax3.set_title(title)
    
    #####################
    ##### w/ DQN ######
    #####################
    
    res = 50


    keys_ = ["MinAtar/SpaceInvaders-v0__dqn__1__1745789971 - charts/local_optimality_gap",
             "MinAtar/SpaceInvaders-v0__dqn__2__1745789970 - charts/local_optimality_gap",
             "MinAtar/SpaceInvaders-v0__dqn__3__1745789971 - charts/local_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='DQN'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    # #####################
    # ##### w/ PPO ######
    # #####################
    
    keys_ = ["MinAtar/SpaceInvaders-v0__ppo__3__1745790012 - charts/local_optimality_gap",
             "MinAtar/SpaceInvaders-v0__ppo__2__1745790012 - charts/local_optimality_gap",
             "MinAtar/SpaceInvaders-v0__ppo__1__1745790012 - charts/local_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='PPO'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label} )
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Local Optimality Gap')
    ax3.set(xlabel='Steps')
    ax3.legend()
    '''
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, ncol=5, mode='expand', bbox_to_anchor=(.37,.03,.3,.1))
    '''
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig(title+".svg")
    fig.savefig(title+".png")
    fig.savefig(title+".pdf")


    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Local Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/MinAtar_Asterix_local_optimality_gap.csv'
    title = 'Local Optimality for Asterix'
    df = pd.read_csv(datadir)
    ax3.set_title(title)
    
    #####################
    ##### w/ DQN ######
    #####################
    
    res = 50


    keys_ = ["MinAtar/Asterix-v0__dqn__1__1745789970 - charts/local_optimality_gap",
             "MinAtar/Asterix-v0__dqn__2__1745789970 - charts/local_optimality_gap",
             "MinAtar/Asterix-v0__dqn__3__1745789970 - charts/local_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='DQN'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    # #####################
    # ##### w/ PPO ######
    # #####################
    
    keys_ = ["MinAtar/Asterix-v0__ppo__1__1745790012 - charts/local_optimality_gap",
             "MinAtar/Asterix-v0__ppo__2__1745790012 - charts/local_optimality_gap",
             "MinAtar/Asterix-v0__ppo__3__1745790012 - charts/local_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='PPO'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label} )
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Local Optimality Gap')
    ax3.set(xlabel='Steps')
    ax3.legend()
    '''
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, ncol=5, mode='expand', bbox_to_anchor=(.37,.03,.3,.1))
    '''
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig(title+".svg")
    fig.savefig(title+".png")
    fig.savefig(title+".pdf")
