import numpy as np
from enum import IntEnum


class BlockType(IntEnum):
    AIR = 0
    STONE = 1
    GRASS = 2
    WOOD = 3
    WATER = 4
    LAVA = 5
    GOAL = 6


BLOCK_COLORS = {
    BlockType.AIR:   (135, 206, 235),
    BlockType.STONE: (128, 128, 128),
    BlockType.GRASS: (76, 153, 0),
    BlockType.WOOD:  (139, 90, 43),
    BlockType.WATER: (64, 164, 223),
    BlockType.LAVA:  (207, 16, 32),
    BlockType.GOAL:  (255, 215, 0),
}

SOLID_BLOCKS = {BlockType.STONE, BlockType.GRASS, BlockType.WOOD}

ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_STRAFE_LEFT = 2
ACTION_STRAFE_RIGHT = 3
ACTION_TURN_LEFT = 4
ACTION_TURN_RIGHT = 5
NUM_ACTIONS = 6

MOVE_SPEED = 0.15
TURN_SPEED = np.pi / 8
GRAVITY = 0.3
AGENT_RADIUS = 0.4
AGENT_HEIGHT = 1.6
EYE_HEIGHT = 1.5


class VoxelWorld:
    def __init__(self, width=16, depth=16, height=8):
        self.width = width
        self.depth = depth
        self.height = height
        self.grid = np.zeros((width, depth, height), dtype=np.uint8)

        self.agent_x = 1.5
        self.agent_y = 1.5
        self.agent_z = 1.0
        self.agent_yaw = 0.0

        self.done = False
        self.reward = 0.0

    def get_block(self, x, y, z):
        ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        if 0 <= ix < self.width and 0 <= iy < self.depth and 0 <= iz < self.height:
            return BlockType(self.grid[ix, iy, iz])
        return BlockType.AIR

    def set_block(self, x, y, z, block_type):
        ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        if 0 <= ix < self.width and 0 <= iy < self.depth and 0 <= iz < self.height:
            self.grid[ix, iy, iz] = int(block_type)

    def is_solid(self, x, y, z):
        return self.get_block(x, y, z) in SOLID_BLOCKS

    def _check_collision(self, x, y, z):
        """
        Checks if a cylinder at (x, y, z).
        """
        min_bx = int(np.floor(x - AGENT_RADIUS))
        max_bx = int(np.floor(x + AGENT_RADIUS))
        min_by = int(np.floor(y - AGENT_RADIUS))
        max_by = int(np.floor(y + AGENT_RADIUS))
        min_bz = int(np.floor(z))
        max_bz = int(np.floor(z + AGENT_HEIGHT))

        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                for bz in range(min_bz, max_bz + 1):
                    if not self.is_solid(bx, by, bz):
                        continue
                    closest_x = np.clip(x, bx, bx + 1)
                    closest_y = np.clip(y, by, by + 1)
                    dx = x - closest_x
                    dy = y - closest_y
                    if dx * dx + dy * dy < AGENT_RADIUS**2:
                        return True
        
        return False

    def _check_special_blocks(self, x, y, z):
        """
        Checks if the agent is standing on or inside lava or goal.
        """
        foot_block = self.get_block(x, y, z)
        standing_on = self.get_block(x, y, z - 0.1)

        for block in (foot_block, standing_on):
            if block == BlockType.GOAL:
                self.reward = 1.0
                self.done = True
                return
            if block == BlockType.LAVA:
                self.reward = -1.0
                self.done = True
                return
    
    def step(self, action):
        self.reward = -0.01
        if self.done:
            return self.reward

        dx, dy = 0.0, 0.0
        forward_x = np.cos(self.agent_yaw)
        forward_y = np.sin(self.agent_yaw)
        right_x = np.cos(self.agent_yaw - np.pi / 2)
        right_y = np.sin(self.agent_yaw - np.pi / 2)

        if action == ACTION_FORWARD:
            dx = forward_x * MOVE_SPEED
            dy = forward_y * MOVE_SPEED
        elif action == ACTION_BACKWARD:
            dx = -forward_x * MOVE_SPEED
            dy = -forward_y * MOVE_SPEED
        elif action == ACTION_STRAFE_LEFT:
            dx = -right_x * MOVE_SPEED
            dy = -right_y * MOVE_SPEED
        elif action == ACTION_STRAFE_RIGHT:
            dx = right_x * MOVE_SPEED
            dy = right_y * MOVE_SPEED
        elif action == ACTION_TURN_LEFT:
            self.agent_yaw += TURN_SPEED
        elif action == ACTION_TURN_RIGHT:
            self.agent_yaw -= TURN_SPEED

        self.agent_yaw = self.agent_yaw % (2 * np.pi)

        # Try X movement
        new_x = self.agent_x + dx
        if not self._check_collision(new_x, self.agent_y, self.agent_z):
            self.agent_x = new_x

        # Try Y movement
        new_y = self.agent_y + dy
        if not self._check_collision(self.agent_x, new_y, self.agent_z):
            self.agent_y = new_y

        # Try Z movement
        new_z = self.agent_z - GRAVITY
        if self._check_collision(self.agent_x, self.agent_y, new_z):
            self.agent_z = np.floor(self.agent_z) if self.agent_z == np.floor(self.agent_z) else self.agent_z
        else:
            self.agent_z = max(new_z, 0.0)

        self._check_special_blocks(self.agent_x, self.agent_y, self.agent_z)
        return self.reward

    def get_agent_eye_pos(self):
        return (self.agent_x, self.agent_y, self.agent_z + EYE_HEIGHT)

    def get_agent_state(self):
        return {
            "x": self.agent_x,
            "y": self.agent_y,
            "z": self.agent_z,
            "yaw": self.agent_yaw
        }

    def build_test_room(self):
        """Create a simple enclosed room for testing."""
        # Floor
        for x in range(self.width):
            for y in range(self.depth):
                self.set_block(x, y, 0, BlockType.GRASS)
        # Walls around the perimeter, 3 blocks tall
        for z in range(1, 4):
            for x in range(self.width):
                self.set_block(x, 0, z, BlockType.STONE)
                self.set_block(x, self.depth - 1, z, BlockType.STONE)
            for y in range(self.depth):
                self.set_block(0, y, z, BlockType.STONE)
                self.set_block(self.width - 1, y, z, BlockType.STONE)
        # A wood pillar in the middle
        for z in range(1, 4):
            self.set_block(8, 8, z, BlockType.WOOD)
        # Lava patch
        self.set_block(5, 5, 0, BlockType.LAVA)
        self.set_block(5, 6, 0, BlockType.LAVA)
        # Goal
        self.set_block(14, 14, 0, BlockType.GOAL)
        # Place agent
        self.agent_x = 2.5
        self.agent_y = 2.5
        self.agent_z = 1.0
        self.agent_yaw = 0.0
        self.done = False