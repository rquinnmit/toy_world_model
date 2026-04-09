import numpy as np
from env.voxel_world import VoxelWorld, BlockType


class BSPNode:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left = None
        self.right = None
        self.room = None

    def split(self, rng, min_size=4):
        if self.left is not None:
            return

        if self.w < min_size * 2 and self.h < min_size * 2:
            return

        if self.w < min_size * 2:
            horizontal = True
        elif self.h < min_size * 2:
            horizontal = False
        else:
            horizontal = rng.random() < 0.5

        if horizontal:
            split_pos = rng.integers(min_size, self.h - min_size + 1)
            self.left = BSPNode(self.x, self.y, self.w, split_pos)
            self.right = BSPNode(self.x, self.y + split_pos, self.w, self.h - split_pos)
        else:
            split_pos = rng.integers(min_size, self.w - min_size + 1)
            self.left = BSPNode(self.x, self.y, split_pos, self.h)
            self.right = BSPNode(self.x + split_pos, self.y, self.w - split_pos, self.h)

        self.left.split(rng, min_size)
        self.right.split(rng, min_size)

    def get_leaves(self):
        if self.left is None:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    def center(self):
        if self.room:
            rx, ry, rw, rh = self.room
            return (rx + rw // 2, ry + rh // 2)
        return (self.x + self.w // 2, self.y + self.h // 2)


def generate_level(world, seed=None):
    rng = np.random.default_rng(seed)
    world.grid.fill(0)

    for x in range(world.width):
        for y in range(world.depth):
            world.set_block(x, y, 0, BlockType.GRASS)

    root = BSPNode(1, 1, world.width - 2, world.depth - 2)
    root.split(rng, min_size=4)
    leaves = root.get_leaves()

    for leaf in leaves:
        margin = 1
        rx = leaf.x + margin
        ry = leaf.y + margin
        rw = max(leaf.w - 2 * margin, 2)
        rh = max(leaf.h - 2 * margin, 2)
        leaf.room = (rx, ry, rw, rh)

        for z in range(1, 4):
            for x in range(rx - 1, rx + rw + 1):
                world.set_block(x, ry - 1, z, BlockType.STONE)
                world.set_block(x, ry + rh, z, BlockType.STONE)
            for y in range(ry - 1, ry + rh + 1):
                world.set_block(rx - 1, y, z, BlockType.STONE)
                world.set_block(rx + rw, y, z, BlockType.STONE)

    _connect_rooms(world, root)

    num_lava = rng.integers(2, 5)
    lava_rooms = rng.choice(len(leaves), size=min(num_lava, len(leaves)), replace=False)
    for idx in lava_rooms:
        rx, ry, rw, rh = leaves[idx].room
        lx = rng.integers(rx, rx + rw)
        ly = rng.integers(ry, ry + rh)
        world.set_block(lx, ly, 0, BlockType.LAVA)

    spawn_room = leaves[0]
    sx, sy = spawn_room.center()
    best_dist = -1
    goal_room = leaves[-1]
    for leaf in leaves:
        cx, cy = leaf.center()
        d = abs(cx - sx) + abs(cy - sy)
        if d > best_dist:
            best_dist = d
            goal_room = leaf

    gx, gy = goal_room.center()
    world.set_block(gx, gy, 0, BlockType.GOAL)

    world.agent_x = sx + 0.5
    world.agent_y = sy + 0.5
    world.agent_z = 1.0
    world.agent_yaw = 0.0
    world.done = False


def _connect_rooms(world, node):
    if node.left is None:
        return

    _connect_rooms(world, node.left)
    _connect_rooms(world, node.right)

    cx1, cy1 = _subtree_center(node.left)
    cx2, cy2 = _subtree_center(node.right)

    x, y = cx1, cy1
    while x != cx2:
        x += 1 if cx2 > x else -1
        _carve_corridor(world, x, y)
    while y != cy2:
        y += 1 if cy2 > y else -1
        _carve_corridor(world, x, y)


def _subtree_center(node):
    if node.left is None:
        return node.center()
    leaves = node.get_leaves()
    cx = sum(l.center()[0] for l in leaves) // len(leaves)
    cy = sum(l.center()[1] for l in leaves) // len(leaves)
    return (cx, cy)


def _carve_corridor(world, x, y):
    for z in range(1, 4):
        world.set_block(x, y, z, BlockType.AIR)
