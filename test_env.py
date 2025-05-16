# import gym
# import crafter

# env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
# env = crafter.Recorder(
#   env, './path/to/logdir',
#   save_stats=True,
#   save_video=False,
#   save_episode=False,
# )

# obs = env.reset()
# done = False
# while not done:
#   action = env.action_space.sample()
#   obs, reward, done, info = env.step(action)

import jax
import craftax
import gymnax

rng = jax.random.key(0)
_rngs, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("Craftax-Symbolic-v1")


# Get an initial state and observation
obs, state = env.reset(_rngs[0], env_params)

# Pick random action
action = env.action_space(env_params).sample(_rngs[1])

# Step environment
obs, state, reward, done, info = env.step(_rngs[2], state, action, env_params)