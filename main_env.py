from RubikCube_env import RubikCube, update_cube_array
import time
import matplotlib.pyplot as plt


size = 2

env = RubikCube(size=size, render_mode="human")    
obs, info = env.reset()


for i in range(200): 
    action = env.action_space.sample()
    time.sleep(0.2)
    obs, reward, done, truncated, info = env.step(action)
    # env.render(mode="ascii")  # Draw after each step
    if done:
        print("done")
        break
