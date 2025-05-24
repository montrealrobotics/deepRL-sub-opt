import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rliable

# Import specific modules
from rlplot import metrics
from rlplot import library as rly
from rlplot import plot_utils
from rlplot.plot_helpers import \
    load_all_exp_data, generate_intervals, generate_pairs, \
    save_metric, get_metric, read_and_norm_algo_scores, save_fig, \
    get_rank_matrix

REPS = 1000
CONFIDENCE = 0.68

# Example performance data
# Each algorithm has scores for multiple runs across multiple tasks
# Shape: (num_runs, num_tasks)
aggregate_names = ['IQM', 'Mean', 'Median', 'Optimality Gap']
scores = {
    'Algorithm1': np.random.uniform(0.0, 1.0, size=(10, 8)),  # 10 runs, 8 tasks
    'Algorithm2': np.random.uniform(0.2, 0.9, size=(10, 8)),
    'Algorithm3': np.random.uniform(0.1, 0.8, size=(10, 8))
}
normalized_algo_scores = scores.copy()

# Get interval estimates for IQM (Interquartile Mean)
aggregate_func_mapper = {
    'Mean': metrics.aggregate_mean,
    'IQM': metrics.aggregate_iqm,
    'Median': metrics.aggregate_median,
    'Optimality Gap': metrics.aggregate_optimality_gap,
}

def aggregate_func(x): return \
    np.array([aggregate_func_mapper[name](x) for name in aggregate_names])

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
    xlabel_y_coordinate=-0.25,
)
# plt.tight_layout()
# plt.show()
save_fig(fig, 'metric_value', "data")