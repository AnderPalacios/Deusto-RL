from RubikCube_env import RubikCube, update_cube_array
import time
import matplotlib.pyplot as plt



env = RubikCube(size=3)    
obs, info = env.reset()
print(update_cube_array(obs, 3))
env.render()


for _ in range(3):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render(mode="ascii")  # Draw after each step
    # env.render(mode="human")
    time.sleep(0.2)
    if done:
        print("done")
        break
