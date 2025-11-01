from RubikCube_env import RubikCube, update_cube_array
import time
import matplotlib.pyplot as plt


size = 2

env = RubikCube(size=size)    
env.verify_consistency()
obs, info = env.reset()
print(obs)
env.render(mode="human")
env.verify_consistency()


# for i in range(20): 
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     env.render(mode="human")  # Draw after each step
#     time.sleep(0.3)
#     if done:
#         print("done")
#         break
