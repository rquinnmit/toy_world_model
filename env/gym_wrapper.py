import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.voxel_world import VoxelWorld, NUM_ACTIONS
from env.renderer import Renderer
from env.procgen import generate_level


class VoxelWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, width=16, depth=16, height=8, render_width=64, render_height=64, max_steps=500, render_mode="rgb_array"):
        super().__init__()
        self.world = VoxelWorld(width, depth, height)
        self.renderer = Renderer(render_width, render_height)
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(render_height, render_width, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._seed = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.default_rng().integers(0, 2**31)

        generate_level(self.world, seed=self._seed)
        self.current_step = 0

        obs = self.renderer.render(self.world)
        info = {"agent_state": self.world.get_agent_state()}
        return obs, info

    def step(self, action):
        reward = self.world.step(action)
        self.current_step += 1

        terminated = self.world.done
        truncated = self.current_step >= self.max_steps

        obs = self.renderer.render(self.world)
        info = {"agent_state": self.world.get_agent_state()}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.renderer.render(self.world)
