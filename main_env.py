from RubikCube_env import RubikCube, update_cube_array
import time
import matplotlib.pyplot as plt


size = 2

env = RubikCube(size=size)    
obs, info = env.reset()
env.render()


for i in range(2): 
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render(mode="ascii")  # Draw after each step
    time.sleep(0.3)
    if done:
        print("done")
        break
