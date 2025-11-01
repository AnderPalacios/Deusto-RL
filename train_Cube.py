import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from RubikCube_env import RubikCube
import numpy as np
import time


def make_env():
    return RubikCube(size=2, difficulty_level=1)


env = DummyVecEnv([make_env])

check_env(env.envs[0])  


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=16,
    n_steps=32,
    learning_rate=1e-3,
    gamma=0.9,
)


model.learn(total_timesteps=5000) 


inner_env = env.envs[0]
inner_env.render(mode="human")
time.sleep(4)
obs, info = inner_env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = inner_env.step(action)
    total_reward += reward
    print(f"Step reward: {reward}")
    inner_env.render(mode="human")
    time.sleep(1)  
