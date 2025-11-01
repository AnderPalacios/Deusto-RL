import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


actionList = {0:'F CW', 1:'U CW', 2:'L CW', 3:'D CW', 4:'R CW', 5:'B CW', 6:'F CCW', 7:'U CCW', 8:'L CCW', 9:'D CCW', 10:'R CCW', 11:'B CCW'}

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

def print_action(action):
    print(f"Action: {actionList[action]}")


def update_cube_array(cube, size):
    coordinates = []
    for face in faces:
        for i in range(size):
            for j in range(size):
                # print(f"Face: {face}, {i,j}, value: {cube[face][i,j]}")
                coordinates.append(cube[face][i,j])
    return np.array(coordinates)



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
        print("         │ {}│ {}│ {}│".format(coords[27], coords[28], coords[29]))   
        print("         ├──┼──┼──┤")
        print("         │ {}│ {}│ {}│".format(coords[30], coords[31], coords[32]))
        print("         ├──┼──┼──┤")
        print("         │ {}│ {}│ {}│".format(coords[33], coords[34], coords[35]))
        print("         └──┴──┴──┘")





class RubikCube(gym.Env):
    metadata = {"render_modes": ["human","ascii"]}

    def __init__(self, size, difficulty_level=1):
        super(RubikCube, self).__init__()
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_xlim(0, size*4)
        self.ax.set_ylim(0, size*3)
        self.rects = []

        self.last_action = None
        self.difficulty = difficulty_level
        self.max_steps = 100
        self.current_step = 0
        self.size = size
        self.action_space = spaces.Discrete(12) # Each face (6) can rotate in 2 ways; Clockwise or Counterclockwise
        self.observation_space = spaces.Box(low=0, high=5, shape=(6*size*size,), dtype=np.int8)
        self.cube = {face: np.zeros((size,size), dtype=int) for face in faces}

    def reset(self, seed=None, options=None):
        self.cube = {face: np.full((self.size, self.size), i, dtype=int) for i, face in enumerate(faces)}
        
        self.current_step = 0
        if self.difficulty == 1:
            self.max_steps = 10
            self.scramble()
        return self.dict_to_array(self.cube), {}
    
    def step(self, action):
        # print_action(action) 
        face_index = action % 6 # ['F CW', 'U CW', 'L CW', 'D CW', 'R CW', 'B CW', 'F CCW', 'U CCW', 'L CCW', 'D CCW', 'R CCW', 'B CCW']
        clockwise = action < 6
        face = faces[face_index] # Face I'll update CW or CCW

        self.rotate_face(face, clockwise)

        ######################## Might change this reward, just trying 
        reward = self.get_reward(action)  # <-- shaped reward

        self.last_action = action
        self.current_step += 1

        done = self.is_solved() or self.current_step >= self.max_steps

        obs_array = self.dict_to_array(self.cube)
        return obs_array, reward, done, False, {}
    

    def is_solved(self):
        for i, face in enumerate(faces):
            if not np.all(self.cube[face] == i):
                return False
        return True
    

    # Possible reward function -> Number of correct tiles
    def get_reward(self, action):
        total_tiles = self.size**2 * 6
        correct_tiles = sum(np.sum(self.cube[face] == i) for i, face in enumerate(faces))
        
        reward = correct_tiles / total_tiles

        # Big reward if the cube is fully solved
        if correct_tiles == total_tiles:
            reward += 100.0
        
        if self.last_action is not None:
            # Check if this action is the opposite of the last
            if action // 6 == self.last_action // 6 and action % 6 != self.last_action % 6:
                reward -= 5.0  # small penalty
        
        # Otherwise, reward is just the number of correct tiles
        return float(correct_tiles)




    def draw_cube_human(self):
        s = self.size
        # first time: create rectangles
        if not self.rects:
            for face, (x_off, y_off) in positions[str(s)].items():
                for i in range(s):
                    for j in range(s):
                        color_idx = self.cube[face][i,j].item()
                        rect = Rectangle((x_off+j, y_off+s-1-i), 1, 1,
                                         facecolor=colors[color_idx], edgecolor='black')
                        self.ax.add_patch(rect)
                        self.rects.append(rect)
            self.fig.canvas.draw()
        else:
            # update colors
            idx = 0
            for face, (x_off, y_off) in positions[str(s)].items():
                for i in range(s):
                    for j in range(s):
                        color_idx = self.cube[face][i,j].item()
                        self.rects[idx].set_facecolor(colors[color_idx])
                        idx += 1
            self.fig.canvas.draw_idle()

        plt.pause(0.1) #0.1


    # Rotation logic     
    def rotate_face(self, face, clockwise=True):
        # Rotate the face itself
        self.cube[face] = np.rot90(self.cube[face], -1 if clockwise else 1)
        s = self.size
        
        # get the 2x2 or 3x3 faces
        F, U, L, D, R, B = [self.cube[f] for f in faces]

        """
        After CW or CCW rotation of the face; update the neighborhood edges
        The four neighboring edges around that face also move in a cycle

        We go face by face updating the 4 adjacent faces got changed. 
        There is of course 1 face where nothing changes (F,B,L,R,U,D are always with respect to F):
        F(green)->B(blue)
        L(orange)->R(red)
        U(white)->D(yellow)
        """
        # FRONT
        if face == 'F':
            if clockwise: 
                temp = U[s-1, :].copy()
                U[s-1, :] = np.flip(L[:, s-1])
                L[:, s-1] = D[0, :]
                D[0, :] = np.flip(R[:, 0])
                R[:, 0] = temp
            else:
                temp = U[s-1, :].copy()
                U[s-1, :] = R[:, 0]
                R[:, 0] = np.flip(D[0, :])
                D[0, :] = L[:, s-1]
                L[:, s-1] = np.flip(temp)

        # BACK
        elif face == 'B':
            if clockwise: 
                temp = U[0, :].copy() 
                U[0, :] = R[:, s-1]
                R[:, s-1] = np.flip(D[s-1, :])
                D[s-1, :] = L[:, 0]
                L[:, 0] = np.flip(temp)
            else: 
                temp = U[0, :].copy() # U's top row woth respect to F
                U[0, :] = L[:, 0]
                L[:, 0] = D[s-1, :]
                D[s-1, :] = R[:, s-1]
                R[:, s-1] = temp # Second column of R with respect to F is equal to U's top row

        # UP
        elif face == 'U':
            if clockwise: 
                temp = B[0, :].copy()
                B[0, :] = L[0, :].copy()
                L[0, :] = F[0, :].copy()
                F[0, :] = R[0, :].copy()
                R[0, :] = temp
            else: 
                temp = B[0, :].copy()
                B[0, :] = R[0, :].copy()
                R[0, :] = F[0, :]
                F[0, :] = L[0, :]
                L[0, :] = temp

        # DOWN
        elif face == 'D':
            if clockwise: 
                temp = F[s-1, :].copy()
                F[s-1, :] = L[s-1, :].copy()
                L[s-1, :] = B[s-1, :].copy()
                B[s-1, :] = R[s-1, :].copy()
                R[s-1, :] = temp
            else: 
                temp = B[s-1, :].copy()
                B[s-1, :] = L[s-1, :].copy()
                L[s-1, :] = F[s-1, :].copy()
                F[s-1, :] = R[s-1, :].copy()
                R[s-1, :] = temp

        # LEFT
        elif face == 'L':
            if clockwise: 
                temp = U[:, 0].copy()
                U[:, 0] = np.flip(B[:, s-1])
                B[:, s-1] = np.flip(D[:, 0])
                D[:, 0] = F[:, 0]
                F[:, 0] = temp
            else:
                temp = U[:, 0].copy()
                U[:, 0] = F[:, 0]
                F[:, 0] = D[:, 0]
                D[:, 0] = np.flip(B[:, s-1])
                B[:, s-1] = np.flip(temp)

        # RIGHT
        elif face == 'R':
            if clockwise: 
                temp = U[:, s-1].copy()
                U[:, s-1] = F[:, s-1]
                F[:, s-1] = D[:, s-1]
                D[:, s-1] = np.flip(B[:, 0])
                B[:, 0] = np.flip(temp)
            else: 
                temp = U[:, s-1].copy()
                U[:, s-1] = np.flip(B[:, 0])
                B[:, 0] = np.flip(D[:, s-1])
                D[:, s-1] = F[:, s-1]
                F[:, s-1] = temp


    def render(self, mode='human'):
        if mode == "human":
            self.draw_cube_human()
        elif mode == "ascii":
            draw_cube_ascii(self.cube, self.size)

    
    def scramble(self):
        if self.difficulty == 1:
            # Initial observation: R CCW → U CCW → B CW
            self.rotate_face('R', clockwise=False)
            # self.rotate_face('U', clockwise=False)
            # self.rotate_face('B', clockwise=True)

    def dict_to_array(self, cube_dict):
        arr = np.zeros((6, self.size, self.size), dtype=np.int8)
        for i, face in enumerate(faces):
            arr[i] = cube_dict[face]
        return arr.flatten()

    def verify_consistency(self, verbose=True):
        """
        Verifies that for every face, performing CW followed by CCW
        returns the cube to its original solved state.
        """
        self.cube = {face: np.full((self.size, self.size), i, dtype=int) for i, face in enumerate(faces)}
        original_state = {f: self.cube[f].copy() for f in faces}
        consistent = True

        def is_solved():
            for f in faces:
                if not np.array_equal(self.cube[f], original_state[f]):
                    return False
            return True

        def reset_cube():
            for f in faces:
                self.cube[f] = original_state[f].copy()

        # Test both directions
        for f in faces:
            # CW then CCW
            self.rotate_face(f, clockwise=True)
            self.rotate_face(f, clockwise=False)
            if not is_solved():
                consistent = False
                if verbose:
                    print(f"Inconsistency found for face {f} (CW→CCW)")
            reset_cube()

            # CCW then CW
            self.rotate_face(f, clockwise=False)
            self.rotate_face(f, clockwise=True)
            if not is_solved():
                consistent = False
                if verbose:
                    print(f"Inconsistency found for face {f} (CCW→CW)")
            reset_cube()

        if consistent:
            if verbose:
                print("All face rotations are internally consistent.")
        else:
            if verbose:
                print("Some rotations cause mismatches.")

        return consistent

