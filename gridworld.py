import gymnasium as gym

def check_inside(x, y, w, h):
    return (0 <= x < w) and (0 <= y < h)

class GridWorld(gym.Env):

    def __init__(self, cfg=None, **kwargs) -> None:
        super(GridWorld, self).__init__(**kwargs)

        self.cfg = cfg
        self.action_space = gym.spaces.Discrete(cfg["num_actions"])
        if cfg["num_actions"] == 4:
            self.dx = [0, 1, 0, -1]
            self.dy = [-1, 0, 1, 0]
        else:
            self.dy = [-1, -1, 0, 1, 1, 1, 0, -1]
            self.dx = [0, 1, 1, 1, 0, -1, -1, -1]
        self.w, self.h = cfg["world_size"][0], cfg["world_size"][1]
        self.observation_space = gym.spaces.Discrete(self.w * self.h)
        self.curr_pos = [self.w//2, 0]
        self.type = cfg["world_type"]
        self.goal = [self.w//2, 7]
        self.max_steps = 60
        self.curr_steps = 0

        if self.type == "A":
            self.obstacles = [[i, 5] for i in range(1, 5)]
        elif self.type == "B":
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.step_func = self.a_step if self.type == "A" else self.b_step

    def reset(self, seed=None, **kwargs):

        self.curr_steps = 0
        self.curr_pos = [self.w//2, 0]
        return self.w * self.curr_pos[0] + self.curr_pos[1], {}
    
    def step(self, action):
        return self.step_func(action)
    
    def a_step(self, action):

        next_pos = [
            self.curr_pos[0] + self.dx[action],
            self.curr_pos[1] + self.dy[action]
        ]
        if check_inside(next_pos[0], next_pos[1], self.w, self.h) and next_pos not in self.obstacles:
            self.prev_pos = self.curr_pos
            self.curr_pos = next_pos

        reward = -1
        done = False
        truncated = False
        self.curr_steps += 1
        if self.curr_pos == self.goal:
            reward = 1
            done = True
        elif self.curr_steps >= self.max_steps:
            truncated = True

        return self.h * self.curr_pos[0] + self.curr_pos[1], reward, done, truncated, {}
    
    def b_step(self, action):

        next_pos = [
            self.curr_pos[0] + self.dy[action],
            self.curr_pos[1] + self.dx[action]
        ]
        next_pos[0] -= self.wind[self.curr_pos[1]]
        if check_inside(next_pos[0], next_pos[1], self.w, self.h):
            self.curr_pos = next_pos

        reward = -1
        done = False
        truncated = False
        self.curr_steps += 1
        if self.curr_pos == self.goal:
            reward = 1
            done = True
        elif self.curr_steps >= self.max_steps:
            truncated = True
        return self.w * self.curr_pos[0] + self.curr_pos[1], reward, done, truncated, {}
