import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from RubikCube_env import RubikCube
import numpy as np
import time


def make_env():
    return RubikCube(size=3, difficulty_level=3, render_mode="None")


env = VecMonitor(DummyVecEnv([make_env]))

#check_env(env.envs[0])  


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=8e-4,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=0,
    device='cpu'

)

# --- entrenamiento por fases ---
phases = [
    (3, 200_000),
    (4, 400_000),
    (5, 600_000),
]

for lvl, steps in phases:
    # fija la dificultad en TODOS los envs vectorizados
    env.set_attr("difficulty", lvl)
    # (opcional) adapta max_steps si quieres más tiempo por episodio en niveles altos
    env.set_attr("max_steps", max(20, 10 + 5*lvl))
    model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=True)


inner_env = env.envs[0]
inner_env.render_mode="human"
time.sleep(4)
obs, info = inner_env.reset()
done = False
truncated = False
total_reward = 0

while not done and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = inner_env.step(action)
    total_reward += reward
    print(f"Step reward: {reward}")
    print("Action taken:", action)
    #inner_env.render(mode="human")
    time.sleep(1)  
