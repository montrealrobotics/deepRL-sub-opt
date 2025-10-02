# DeepRL Sub-optimality 

Code for the paper on analyzing the sub-optimality of deep RL algorithms. Based on the cleanrl codebase.

# classic control
python cleanrl/dqn.py --env-id CartPole-v1
python cleanrl/ppo.py --env-id CartPole-v1
python cleanrl/c51.py --env-id CartPole-v1

# atari
poetry install -E atari
python cleanrl/dqn_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/c51_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/sac_atari.py --env-id BreakoutNoFrameskip-v4


## Citing Paper

If you use CleanRL in your work, please cite our technical [paper](https://www.jmlr.org/papers/v23/21-1342.html):

```bibtex
@misc{berseth2025explorationoptimizationproblemdeep,
      title={Is Exploration or Optimization the Problem for Deep Reinforcement Learning?}, 
      author={Glen Berseth},
      year={2025},
      eprint={2508.01329},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.01329}, 
}
```
