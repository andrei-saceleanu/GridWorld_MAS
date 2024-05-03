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
            self.dx = [0, 1, 1, 1, 0, -1, -1, -1]
            self.dy = [-1, -1, 0, 1, 1, 1, 0, -1]
            
        self.h, self.w = cfg["world_size"][0], cfg["world_size"][1]
        self.observation_space = gym.spaces.Discrete(self.w * self.h)
        self.start_states = [
            [0, self.h//2],
            [2, self.h-1],
            [self.w, self.h//2-1]
        ]
        self.start_state_idx = cfg["start_state_idx"]
        self.cfg = cfg
        if cfg["task3"]:
            self.curr_pos = [self.start_states[self.start_state_idx] for _ in range(3)]
        else:
            self.curr_pos = self.start_states[self.start_state_idx]
        self.type = cfg["world_type"]
        self.goal = [7, self.h//2]
        self.curr_steps = 0

        if self.type == "A":
            self.obstacles = [[5, i] for i in range(1, 5)]
            self.max_steps = 200 
        elif self.type == "B":
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
            self.max_steps = 400 

        self.start_states

        self.step_func = self.a_step if self.type == "A" else self.b_step

    def reset(self, seed=None, **kwargs):

        self.curr_steps = 0
        if self.cfg["task3"]:
            self.curr_pos = [self.start_states[self.start_state_idx] for _ in range(3)]
        else:
            self.curr_pos = self.start_states[self.start_state_idx]
        return self.w * self.curr_pos[1] + self.curr_pos[0], {}
    
    def step(self, action):
        return self.step_func(action)
    
    def a_step(self, action):
        if self.cfg["task3"]:
            reward = [-1, -1, -1]
            done = [False, False, False]
            finishedNow = 0
            for i in range(len(action)):
                if self.curr_pos[i] == self.goal:
                    reward[i] = 0
                    done[i] = True
                    continue
                next_pos = [
                    self.curr_pos[i][0] + self.dx[action[i]],
                    self.curr_pos[i][1] + self.dy[action[i]]
                ]
                if check_inside(next_pos[0], next_pos[1], self.w, self.h) and next_pos not in self.obstacles:
                    # self.prev_pos[i] = self.curr_pos[i]
                    self.curr_pos[i] = next_pos

                # done = False
                truncated = False
                self.curr_steps += 1
                
                if self.curr_pos[i] == self.goal:
                    reward[i] = 0.5
                    done[i] = True
                    finishedNow += 1
                elif self.curr_steps >= self.max_steps:
                    truncated = True
            if finishedNow == len(action):
                reward = [10, 10, 10]
            return [self.w * self.curr_pos[i][1] + self.curr_pos[i][0] for i in range(len(action))], reward, done, truncated, {}
        else:
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

            return self.w * self.curr_pos[1] + self.curr_pos[0], reward, done, truncated, {}
    
    def b_step(self, action):
        next_pos = [
            self.curr_pos[0] + self.dx[action],
            self.curr_pos[1] + self.dy[action]
        ]
        if check_inside(next_pos[0], next_pos[1], self.w, self.h):
            next_pos[1] -= self.wind[next_pos[0]]
            next_pos[1] = max(0, next_pos[1])
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
        return self.w * self.curr_pos[1] + self.curr_pos[0], reward, done, truncated, {}
