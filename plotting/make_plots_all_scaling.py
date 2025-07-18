

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    # res = 50
    # lw_ = 3
    # max_ = 290000000
    # fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    # datadir = './data/DQN_BattleZone_ResNet2.csv'
    # df = pd.read_csv(datadir)
    # title = 'Global Optimality Gap on BattleZone Scaling'
    # ax3.set_title(title)

    # jobs = get_jobs(df)


    # #####################
    # ##### w/ \pi ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w ResNet'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    # #####################
    # ##### w/ \pi deterministic ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs)

    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w ResNet'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # datadir = './data/DQN_BattleZone.csv'
    # df = pd.read_csv(datadir)
    # ax3.set_title(title)

    # jobs = get_jobs(df)

    # #####################
    # ##### w/ \pi ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    # #####################
    # ##### w/ \pi deterministic ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    # ax3.set(ylabel='Return')
    # ax3.set(xlabel='Steps')
    # ax3.legend(fontsize='14')
    # # fig.tight_layout(pad=0.5)
    # #plt.subplots_adjust(bottom=.25, wspace=.25)
    # plt.show()
    # fig.savefig("data/"+title+".svg")
    # fig.savefig("data/"+title+".png")
    # fig.savefig("data/"+title+".pdf")



    res = 50
    lw_ = 3
    max_ = 2200000
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/DQN_NameThisGame_ResNet2.csv'
    df = pd.read_csv(datadir)
    title = 'Difference on NameThisGame Scaling'
    ax3.set_title(title)

    jobs = get_jobs(df)


    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w ResNet'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs)

    label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w ResNet'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/DQN_NameThisGame.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")