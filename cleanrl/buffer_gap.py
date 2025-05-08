


## A special buffer to help track the optimality gap between data generated
from typing import Any, Dict, Generator, List, Optional, Union
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces
import numpy as np
import torch as th

class BufferGap(ReplayBuffer):
    """
    A special buffer to help track the optimality gap between data generated
    by the agent and the optimal policy.
    """

    def __init__(self, 
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        *args, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,
                         optimize_memory_usage, handle_timeout_termination)
        # super().__init__(*args, **kwargs)
        self.gap = 0.0
        self._gap_percentage = 0.05
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        ## Check is the info dictionary contains the return key "r" if it does add that to the returns vector
        
        self.returns[self.pos] = np.array([info.get("return") for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)

import heapq, collections, torch, gym

class BufferGapV2():

    def __init__(self, 
        buffer_size: int,
        top_buffer_percet: float = 0.05,
        policy = False, 
        device = "auto", 
        args = None,
        envs = None,
    ):
        
        self._max_return = -100000
    ## max_returns is a list of the top 10 episodic returns
        self._max_returns = []
        self._max_return_buff_size = buffer_size * (top_buffer_percet)
        self._top_buffer_percet = top_buffer_percet
        ## returns is a deque of the last 100 episodic returns
        self._returns = collections.deque(maxlen=buffer_size)

        self._policy = policy
        self._device = device

        # env setup
        self._envs = envs
        self._args = args
        self._last_eval = 0


    def add(self, return_: float):
        self._returns.append(return_)
    
        if return_ > self._max_return:
            self._max_return = return_
        if len(self._max_returns) == 0:
            ## If this is the first return jsut story that return
            self._max_returns = [return_]
            # heapq.heapify(max_returns)
        if len(self._max_returns) > 0 and return_ > min(self._max_returns):
            ## Repalce the minimum value in max_returns with the new episodic return if the buffer is full
            if (len(self._max_returns) < self._max_return_buff_size ):
                ## If the buffer is not full, just append the new episodic return
                self._max_returns.append(return_)
                heapq.heapify(self._max_returns)
            else:
                heapq.heapreplace(self._max_returns, return_)


    def plot_gap(self, writer, step: int):
        """
        Plot the gap between the current return and the maximum return
        """
        returns_ = list(self._returns)
        heapq.heapify(returns_)
        writer.add_scalar("charts/best_trajectory_return", self._max_return, step)
        writer.add_scalar("charts/avg_top_returns_global", np.mean(list(self._max_returns)), step)
        writer.add_scalar("charts/avg_top_returns_local", np.mean(heapq.nlargest(max(int(self._top_buffer_percet * len(returns_)), 1), returns_)), step)
        writer.add_scalar("charts/global_optimality_gap", np.mean(list(self._max_returns)) - np.mean(returns_), step)
        writer.add_scalar("charts/local_optimality_gap", np.mean(heapq.nlargest(max(int(self._top_buffer_percet * len(returns_)), 1), returns_)) - np.mean(returns_), step)
        
        ## Get performance for the deterministic policy
        if step - self._last_eval > 10000:
            returns = self.eval_deterministic()
            self._last_eval = step
            writer.add_scalar("charts/deterministic returns", np.mean(returns), step)


    def eval_deterministic(self) -> np.ndarray:
        """
        Evaluate the policy deterministically
        """
        self._envs.reset(seed=self._args.seed)
        # q_values = self._policy(torch.Tensor(obs).to(self._device))
        # actions = torch.argmax(q_values, dim=1).cpu().numpy()
        returns = []
        t=0
        max_t = 1000
        while t < max_t:
            obs, _ = self._envs.reset()
            return_ = 0.0
            for i in range(max_t):
                
                actions = self._policy.get_action_deterministic(torch.Tensor(obs).to(self._device)).cpu().numpy()
                
                obs, rewards, terminations, truncations, infos = self._envs.step(actions)
                
                return_ += rewards
                t += 1
                # Check if the episode is done
                if terminations.any() or truncations.any():
                    break
            returns.append(return_)

        return np.mean(returns)

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminations, truncations, infos = self._envs.step(actions)
        # return_ += rewards
        # infos["return"] = return_

        