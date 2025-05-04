

"""
sudo pip3 install pandas
sudo pip3 install seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)
import pdb

colors = {'SMiRL (ours)': 'k',
          'SMiRL VAE (ours)': 'purple',
          'ICM': 'b',
          'RND': 'orange',
          'Oracle': 'g',
          'Reward + SMiRL (ours)' : 'k',
          'Reward + ICM' : 'b',
          'Reward': 'r',
          'SMiRL + ICM': 'brown',
         }
linestyle = {'SMiRL (ours)': '-',
          'ICM': '-',
          'RND': '--',
          'Oracle': '--',
          'SMiRL VAE (ours)': '--',
          'Reward + SMiRL (ours)' : '-',
          'Reward + ICM' : '-',
          'Reward': '--',
          'SMiRL + ICM': '-',
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
        
if __name__ == '__main__':

    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Stability #####
    #####################
    #*******************************************************************************
    
    datadir = './data/SpaceInvaders_CNN.csv'
    df = pd.read_csv(datadir)
    ax3.set_title('Optimality for SpaceInvaders')
    
    #####################
    ##### w/ SMiRL ######
    #####################
    
    res = 5
    steps_ = df["Step"].to_numpy()
    data_ = df["SpaceInvadersNoFrameskip-v4__dqn_atari__1__1745790578 - charts/local_optimality_gap"].to_numpy()
    
    data_ = (np.cumsum(data_)[res:] - np.cumsum(data_)[:-res])/res
    plot_data = [(step_, val) for step_, val in zip(steps_, data_)]
    
    
    data_ = df["SpaceInvadersNoFrameskip-v4__dqn_atari__3__1745790573 - charts/local_optimality_gap"].to_numpy()
    data_ = (np.cumsum(data_)[res:] - np.cumsum(data_)[:-res])/res
    plot_data.extend([(step_, val) for step_, val in zip(steps_, data_)])
    
    bf = pd.DataFrame(plot_data)
    label='local_optimality_gap'
    bf = bf.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=bf, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    # #####################
    # ##### w/ ICM ######
    # #####################
    
    steps_ = df["Step"].to_numpy()
    data_ = df["SpaceInvadersNoFrameskip-v4__dqn_atari__1__1745790578 - charts/local_optimality_gap"].to_numpy()
    
    data_ = (np.cumsum(data_)[res:] - np.cumsum(data_)[:-res])/res
    plot_data = [(step_, val) for step_, val in zip(steps_, data_)]
    
    
    data_ = df["SpaceInvadersNoFrameskip-v4__dqn_atari__2__1745790573 - charts/local_optimality_gap"].to_numpy()
    data_ = (np.cumsum(data_)[res:] - np.cumsum(data_)[:-res])/res
    plot_data.extend([(step_, val) for step_, val in zip(steps_, data_)])

    
    bf = pd.DataFrame(plot_data)
    label='optimality_gap_Max'
    bf = bf.rename(columns={0: 'Steps', 1: label} )
    sns.lineplot(data=bf, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='% Optimality Gap')
    ax3.set(xlabel='Steps')
    ax3.legend()
    '''
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, ncol=5, mode='expand', bbox_to_anchor=(.37,.03,.3,.1))
    '''
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("file0"+".svg")
    fig.savefig("file0"+".png")
