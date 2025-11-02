import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from RubikCube_env import RubikCube
import numpy as np
import time


def make_env():
    return RubikCube(size=3, difficulty_level=1, render_mode="None")


env = VecMonitor(DummyVecEnv([make_env]))

#check_env(env.envs[0])  


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,          # 1 env -> 2048 muestras por actualización
    batch_size=256,        # divisor de 2048
    n_epochs=10,           # típicamente 10
    learning_rate=8e-4,  # estable para PPO
    gamma=0.995,           # horizonte largo
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # un poco de exploración
    vf_coef=0.5,
    max_grad_norm=0.5,       # evita pasos demasiado grandes
    seed=0,
    device='cpu'

)



model.learn(total_timesteps=35000) # 100000


inner_env = env.envs[0]
inner_env.render_mode="human"
time.sleep(4)
obs, info = inner_env.reset()
done = False
truncated = False
total_reward = 0

while not done and not truncated:
    time.sleep(1.5)  
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = inner_env.step(action)
    total_reward += reward
    print(f"Step reward: {reward}")
    print("Action taken:", action)
    #inner_env.render(mode="human")
    if done:
        time.sleep(2)