import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


actionList = ['F CW', 'R CW', 'L CW', 'U CW', 'D CW', 'B CW', 'F CCW', 'R CCW', 'L CCW', 'U CCW', 'D CCW', 'B CCW']

tileDict = {'R' : 0, 'W' : 1, 'G' : 2, 'Y' : 3, 'B' : 4, 'O' : 5}

faces = ['F', 'U', 'L', 'D', 'R', 'B']

colors = ['green', 'white', 'orange', 'yellow', 'red', 'blue']

positions = {
    '2':{
    'U': (2,4),  # Up
    'L': (0,2),  # Left
    'F': (2,2),  # Front
    'R': (4,2),  # Right
    'B': (6,2),  # Back
    'D': (2,0)   # Down
    }, 
    '3': {
    'U': (3,6),
    'L': (0,3),
    'F': (3,3),
    'R': (6,3),
    'B': (9,3),
    'D': (3,0)
}}


def update_cube_array(cube, size):
    coordinates = []
    for face in faces:
        for i in range(size):
            for j in range(size):
                coordinates.append(cube[face][i,j])
    return np.array(coordinates)

def draw_cube_human(cube, size):

    if size != 2 and size != 3:
        raise Exception("Invalid cube size. Must be 2x2 or 3x3")
    
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_xlim(0, size*4+0.1)
    ax.set_ylim(0, size*3)
    ax.set_aspect('equal')
    plt.axis('off')

    for face, (x_off, y_off) in positions.get(str(size)).items():
        for i in range(size):
            for j in range(size):
                color_idx = cube[face][i,j].item()
                rect = Rectangle((x_off+j, y_off+size-1-i), 1,1, facecolor=colors[color_idx], edgecolor='black')
                ax.add_patch(rect)

    plt.show()


"""
       ┌──┬──┐
       │ 4│ 5│
       ├──┼──┤
       │ 6│ 7│
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │ 8│ 9│ 0│ 1│16│17│20│21│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │10│11│ 2│ 3│18│19│22│23│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │12│13│
       ├──┼──┤
       │14│15│
       └──┴──┘

face colors:
    ┌──┐
    │ 1│
 ┌──┼──┼──┬──┐
 │ 2│ 0│ 4│ 5│
 └──┼──┼──┴──┘
    │ 3│
    └──┘
"""


def draw_cube_ascii(cube, size):
    coords = update_cube_array(cube, size)
    if size == 2:
        print("      ┌──┬──┐")
        print("      │ {}│ {}│".format(coords[4], coords[5]))
        print("      ├──┼──┤")
        print("      │ {}│ {}│".format(coords[6], coords[7]))
        print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(coords[8], coords[9], coords[0], coords[1], coords[16], coords[17], coords[20], coords[21]))
        print("├──┼──┼──┼──┼──┼──┼──┼──┤")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(coords[10], coords[11], coords[2], coords[3], coords[18], coords[19], coords[22], coords[23]))
        print("└──┴──┼──┼──┼──┴──┴──┴──┘")
        print("      │ {}│ {}│".format(coords[12], coords[13]))
        print("      ├──┼──┤")
        print("      │ {}│ {}│".format(coords[14], coords[15]))
        print("      └──┴──┘")

    elif size == 3:
        print("         ┌──┬──┬──┐")
        print("         │ {}│ {}│ {}│".format(coords[9], coords[10], coords[11]))   # Up
        print("         ├──┼──┼──┤")
        print("         │ {}│ {}│ {}│".format(coords[12], coords[13], coords[14]))
        print("         ├──┼──┼──┤")
        print("         │ {}│ {}│ {}│".format(coords[15], coords[16], coords[17]))
        print("┌──┬──┬──┼──┼──┼──┼──┬──┬──┬──┬──┬──┐")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(
            coords[18], coords[19], coords[20],   # Left
            coords[0], coords[1], coords[2],     # Front
            coords[36], coords[37], coords[38],  # Right
            coords[45], coords[46], coords[47]   # Back (top row)
        ))
        print("┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(
            coords[21], coords[22], coords[23],   # Left
            coords[3], coords[4], coords[5],     # Front
            coords[39], coords[40], coords[41],  # Right
            coords[48], coords[49], coords[50]   # Back (middle row)
        ))
        print("┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(
            coords[24], coords[25], coords[26],   # Left
            coords[6], coords[7], coords[8],     # Front
            coords[42], coords[43], coords[44],  # Right
            coords[51], coords[52], coords[53]   # Back (bottom row)
        ))
        print("└──┴──┴──┼──┼──┼──┼──┴──┴──┴──┴──┴──┘")
        print("         │ {}│ {}│ {}│".format(coords[27], coords[28], coords[29]))   # Down
        print("         ├──┼──┼──┤")
        print("         │ {}│ {}│ {}│".format(coords[30], coords[31], coords[32]))
        print("         ├──┼──┼──┤")
        print("         │ {}│ {}│ {}│".format(coords[33], coords[34], coords[35]))
        print("         └──┴──┴──┘")





class RubikCube(gym.Env):
    metadata = {"render_modes": ["human","ascii"]}

    def __init__(self, size):
        super(RubikCube, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(12) # Each face (6) can rotate in 2 ways; Clockwise or Counterclockwise
        if size == 2:
            self.observation_space = spaces.Box(low=0, high=5, shape=(6,2,2), dtype=np.int8)
            self.cube = {face: np.zeros((2,2), dtype=int) for face in faces}
        elif size == 3:
            self.observation_space = spaces.Box(low=0, high=5, shape=(6,3,3), dtype=np.int8)
            self.cube = {face: np.zeros((3,3), dtype=int) for face in faces}

    def reset(self, seed=None, options=None):
        self.cube = {face: np.full((self.size, self.size), i, dtype=int) for i, face in enumerate(faces)}
        return self.cube, {}
    
    def step(self, action):
        # Yet to implement rotation logic
        reward = 0
        done = False
        return self.cube, reward, done, False, {}
    
    def render(self, mode='human'):
        if mode == "human":
            draw_cube_human(self.cube, self.size)
        elif mode == "ascii":
            draw_cube_ascii(self.cube, self.size)



    

    


