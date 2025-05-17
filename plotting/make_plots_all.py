

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

colors = {'Global Gap': 'k',
          'ICM': 'b',
          'RND': 'orange',
          'Expert': 'g',
          'Recent top data' : 'k',
          '$\pi$' : 'b',
          '\pi deterministic': 'purple',
          'Best top data': 'brown',
         }
linestyle = {'Global Gap': '-',
          'ICM': '-',
          'RND': '-',
          'Expert': '-',
          'Recent top data': '-',
          'Best top data' : '-',
          'Reward + ICM' : '-',
          '$\pi$': '-',
          '\pi deterministic': '-',
         }
def plotsns_smoothed(ax, s, df, label, title=None, ylabel=None, res=1):
    data = list(df[s])
    s = s.split('/')[-1]
    data = pd.DataFrame([(i//res*res, data[i]) for i in range(len(data))])
    data = data.rename(columns={1: s, 0: 'Episodes'})
    ax = sns.lineplot(x='Episodes', y=s, data=data, label=label, ax=ax, legend=False,  c=colors[label])

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(s)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

def plotsns(ax, s, df, label, title=None, ylabel=None, res=1):
    data = list(df[s])
    data = (np.cumsum(data)[res:]-np.cumsum(data)[:-res]) / res
    s = s.split('/')[-1]
    data = pd.DataFrame([(i, data[i]) for i in range(len(data))])
    data = data.rename(columns={1: s, 0: 'Episodes'})
    ax = sns.lineplot(x='Episodes', y=s, data=data, label=label, ax=ax, legend=False, c=colors[label])
    #print(ax.lines)
    ax.lines[-1].set_linestyle(linestyle[label])
    #print(\label\, label,linestyle[label] )
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(s)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    ax.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    #ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)

def save(fname):
    plt.show()
    '''
    plt.savefig('{}.png'.format(fname))
    plt.clf()
    '''

def deNan(data_):
    if np.isnan(data_[0]):
        ## Search foreward for a value and replace the first values
        i = 0
        while np.isnan(data_[i]):
            i += 1
        data_[:i] = data_[i]

    ## Repalce all NaN values with the last value
    for i in range(len(data_)): 
        if np.isnan(data_[i]):
            data_[i] = data_[i-1]
    return data_

def get_data_frame(df, key, res=10, jobs=None):
    

    plot_data = []
    for i in range(len(jobs)): 
        key__ =   jobs[i]+" - global_step"
        # steps_ = deNan(df[key__].to_numpy())
        steps_ = range(len(df[key__]))
        steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
        data_ = df[jobs[i]+key].to_numpy()
        data_ = deNan(data_)
        print(jobs[i]+key, data_)
        # data_ = (np.cumsum(data_)[res:] - np.cumsum(data_)[:-res])/res
        
        ## Average over the last 5 values with the resulting array being one 5th shorter
        data_ = np.array([np.mean(data_[i:i+res]) for i in range(0, len(data_)-res+1, res)])
        # stds_ = np.array([np.std(data_[i:i+res]) for i in range(0, len(data_)-res+1, res)])


        # plot_data.extend([(step_, val, std_) for step_, val, std_ in zip(steps_, data_, stds_)])
        plot_data.extend([(step_, val) for step_, val in zip(steps_, data_)])
    
    plot_data = pd.DataFrame(plot_data)
    
    return plot_data
        
if __name__ == '__main__':

    res = 20
    lw_ = 3
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/PPO_MR-all.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality for Montezumas Revenge'
    ax3.set_title(title)

    jobs = [
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__0__1747414034",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__6__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__5__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__4__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__3__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__1__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__0__1747414032"
    ]

    #####################
    ##### w/ Optimal ######

    # steps_ = deNan(df["MontezumaRevengeNoFrameskip-v4__ppo_atari__0__1__1747414032 - global_step"].to_numpy())

    # steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
    # plot_data = pd.DataFrame([(step_, 50) for step_ in steps_])

    # label='Expert'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label])
    # ax3.lines[-1].set_linestyle(linestyle[label])

    
    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)

    label='$\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/deterministic returns", res=res, jobs=jobs)

    label='\pi deterministic'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ Global Gap ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='Best top data'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    #####################
    ##### w/ Local Gap ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='Recent top data'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps x $10^4$')
    ax3.legend()
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig(title+".svg")
    fig.savefig(title+".png")
    fig.savefig(title+".pdf")


    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/PPO-halfCheetah.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality for HalfCheetah'
    ax3.set_title(title)

    jobs = [
        "HalfCheetah-v4__ppo_continuous_action__1__1747414032",
        "HalfCheetah-v4__ppo_continuous_action__2__1747414032",
        "HalfCheetah-v4__ppo_continuous_action__3__1747414032",
        "HalfCheetah-v4__ppo_continuous_action__4__1747414032",
    ]

    #####################
    ##### w/ Optimal ######

    # steps_ = deNan(df["MontezumaRevengeNoFrameskip-v4__ppo_atari__0__1__1747414032 - global_step"].to_numpy())

    # steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
    # plot_data = pd.DataFrame([(step_, 50) for step_ in steps_])

    # label='Expert'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label])
    # ax3.lines[-1].set_linestyle(linestyle[label])

    
    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)

    label='$\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/deterministic returns", res=res, jobs=jobs)

    label='\pi deterministic'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ Global Gap ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='Best top data'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    #####################
    ##### w/ Local Gap ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='Recent top data'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps x $10^4$')
    ax3.legend()
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig(title+".svg")
    fig.savefig(title+".png")
    fig.savefig(title+".pdf")
