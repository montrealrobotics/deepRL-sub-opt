from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import numpy as np
import pandas as pd

# Import specific modules
from rlplot import metrics
from rlplot import library as rly
from rlplot import plot_utils
from rlplot.plot_helpers import \
    load_all_exp_data, generate_intervals, generate_pairs, \
    save_metric, get_metric, read_and_norm_algo_scores, save_fig, \
    get_rank_matrix


# from make_plots import *
REPS = 1000
CONFIDENCE = 0.68

# Example performance data
# Each algorithm has scores for multiple runs across multiple tasks
# Shape: (num_runs, num_tasks)
aggregate_names = ['IQM', 'Mean', 'Median', 'Optimality Gap']

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

def get_data_frame(df, key, res=10, jobs=None, max=10000000000):
    

    plot_data = []
    min_ = []
    for i in range(len(jobs)): 
        data_ = df[jobs[i]+key][:max].to_numpy()
        data_ = deNan(data_)
        print(jobs[i]+key, data_)

        # plot_data.extend([(step_, val, std_) for step_, val, std_ in zip(steps_, data_, stds_)])
        print("key", jobs[i]+key, "data_:", data_)
        plot_data.append(np.mean(data_))
        min_.append(np.min(data_))  
    
    return (np.array(plot_data), np.array(min_))

## This function will process the data from a csv file, checking the colum keys and return the strings for those keys
def get_jobs(df):
    keys = []
    for i in range(len(df.columns)):
        key = df.columns[i]
        if ' - charts/global_optimality_gap' in key and "__MIN" not in key and "__MAX" not in key and (len(df[key]) > 10):
            #remove the end of the key
            key_ = key.split(' - charts/global_optimality_gap')[0]
            keys.append(key_)
    return keys
        
if __name__ == '__main__':


    # Get interval estimates for IQM (Interquartile Mean)
    aggregate_func_mapper = {
        'Mean': metrics.aggregate_mean,
        'IQM': metrics.aggregate_iqm,
        'Median': metrics.aggregate_median,
        'Optimality Gap': metrics.aggregate_optimality_gap,
    }

    def aggregate_func(x): return \
        np.array([aggregate_func_mapper[name](x) for name in aggregate_names])
    
    scores = {}
    # x=[]
    datadir = './data/DQN_NameThisGame_ResNet.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
    # x.append(scores)
    (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
    scores['DQN'] = [(scores_ - min_)/(scores_max - min_)]
    datadirs = [
        './data/DQN_Asterix.csv',
        './data/DQN_BattleZone.csv',
        './data/DQN_SpaceInvaders.csv',
        './data/DQN_NameThisGame_ResNet.csv',
        './data/DQN_Minatar_Breakout_all.csv',
    ]
    for datadir in datadirs:
        df = pd.read_csv(datadir)
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        # x.append(scores)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
        scores['DQN'].append(((scores_ - min_)/(scores_max - min_))[:len(scores['DQN'][0])])
    

    datadir = './data/PPO_MR_All.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
    # x.append(scores_)
    (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
    scores['PPO'] = [(scores_ - min_)/(scores_max - min_)]

    datadirs = [
        './data/PPO_Asterix_with_RND.csv',
        './data/PPO_SpaceInvaders.csv',
        './data/PPO_MinAtar_SpaceInvaders.csv',
        './data/PPO_HalfCheetah_4_layers.csv',
        './data/PPO_MR_all2_without_RND.csv',
    ]
    
    for datadir in datadirs:
        df = pd.read_csv(datadir)
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        # x.append(scores_)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
        scores['PPO'].append(((scores_ - min_)/(scores_max - min_))[:len(scores['PPO'][0])])



    for key in scores.keys():
        scores[key] = np.array(scores[key])
    normalized_algo_scores = scores.copy()


    aggregate_scores, aggregate_score_cis = \
        rly.get_interval_estimates(
            normalized_algo_scores, aggregate_func,
            reps=REPS, confidence_interval_size=CONFIDENCE
        )
    # Plot the interval estimates for different metrics
    algos = list(scores.keys())
    # point_estimates = [iqm_scores, mean_scores, median_scores, optimal_scores]
    # interval_estimates = [iqm_cis, mean_cis, median_cis, optimal_cis]

    # print("Point Estimates:", point_estimates)
    # print("Interval Estimates:", interval_estimates)
    # print("Metric Names:", metric_names)
    # print("Algorithms:", algorithms)
    print("Aggregate Scores:", aggregate_scores)

    fig, axes = plot_utils.plot_metric_value(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=aggregate_names,
        algorithms=algos,
        milestone='1m',
        xlabel='Normalized Score',
        xlabel_y_coordinate=-0.65,
    )
    # plt.tight_layout()
    # plt.show()
    save_fig(fig, 'metric_value_gap', "data")

    scores = {}
    # x=[]
    datadir = './data/DQN_NameThisGame_ResNet.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
    # x.append(scores)
    (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_local", jobs=jobs)
    scores['DQN'] = [(scores_ - min_)/(scores_max - min_)]
    datadirs = [
        './data/DQN_Asterix.csv',
        './data/DQN_BattleZone.csv',
        './data/DQN_SpaceInvaders.csv',
        './data/DQN_NameThisGame_ResNet.csv',
        './data/DQN_Minatar_Breakout_all.csv',
    ]
    for datadir in datadirs:
        df = pd.read_csv(datadir)
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        # x.append(scores)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_local", jobs=jobs)
        scores['DQN'].append(((scores_ - min_)/(scores_max - min_))[:len(scores['DQN'][0])])
    

    datadir = './data/PPO_MR_All.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
    # x.append(scores_)
    (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_local", jobs=jobs)
    scores['PPO'] = [(scores_ - min_)/(scores_max - min_)]

    datadirs = [
        './data/PPO_Asterix_with_RND.csv',
        './data/PPO_SpaceInvaders.csv',
        './data/PPO_MinAtar_SpaceInvaders.csv',
        './data/PPO_HalfCheetah_4_layers.csv',
        './data/PPO_MR_all2_without_RND.csv',
    ]
    
    for datadir in datadirs:
        df = pd.read_csv(datadir)
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        # x.append(scores_)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_local", jobs=jobs)
        scores['PPO'].append(((scores_ - min_)/(scores_max - min_))[:len(scores['PPO'][0])])



    for key in scores.keys():
        scores[key] = np.array(scores[key])
    normalized_algo_scores = scores.copy()


    aggregate_scores, aggregate_score_cis = \
        rly.get_interval_estimates(
            normalized_algo_scores, aggregate_func,
            reps=REPS, confidence_interval_size=CONFIDENCE
        )
    # Plot the interval estimates for different metrics
    algos = list(scores.keys())
    # point_estimates = [iqm_scores, mean_scores, median_scores, optimal_scores]
    # interval_estimates = [iqm_cis, mean_cis, median_cis, optimal_cis]

    # print("Point Estimates:", point_estimates)
    # print("Interval Estimates:", interval_estimates)
    # print("Metric Names:", metric_names)
    # print("Algorithms:", algorithms)
    print("Aggregate Scores:", aggregate_scores)

    fig, axes = plot_utils.plot_metric_value(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=aggregate_names,
        algorithms=algos,
        milestone='1m',
        xlabel='Normalized Score',
        xlabel_y_coordinate=-0.65,
    )
    # plt.tight_layout()
    # plt.show()
    save_fig(fig, 'metric_value_gap_recent', "data")