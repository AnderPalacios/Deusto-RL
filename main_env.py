from RubikCube_env import RubikCube, update_cube_array
import time
import matplotlib.pyplot as plt


size = 3

env = RubikCube(size=size, render_mode="human")    
obs, info = env.reset()


for i in range(100): 
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render(render_mode="ascii")  # Draw after each step
    time.sleep(0.3)
    if done:
        print("done")
        break
