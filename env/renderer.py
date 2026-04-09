import numpy as np
from env.voxel_world import BLOCK_COLORS, BlockType


class Renderer:
    def __init__(self, width=64, height=64, fov=np.pi / 3, max_dist=30.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.max_dist = max_dist

    def render(self, world):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        sky_color = np.array(BLOCK_COLORS[BlockType.AIR], dtype=np.uint8)

        eye_x, eye_y, eye_z = world.get_agent_eye_pos()
        yaw = world.agent_yaw

        for px in range(self.width):
            for py in range(self.height):
                # Ray direction
                angle_h = yaw + self.fov * (0.5 - px / self.width)
                angle_v = self.fov * (0.5 - py / self.height)

                ray_dx = np.cos(angle_h) * np.cos(angle_v)
                ray_dy = np.sin(angle_h) * np.cos(angle_v)
                ray_dz = -np.sin(angle_v)

                color, hit = self._cast_ray(world, eye_x, eye_y, eye_z, ray_dx, ray_dy, ray_dz)

                if hit:
                    image[py, px] = color
                else:
                    image[py, px] = sky_color

        return image

    def _cast_ray(self, world, ox, oy, oz, dx, dy, dz):
        """3D DDA raycasting. Returns (color, hit)."""
        # Current voxel
        vx = int(np.floor(ox))
        vy = int(np.floor(oy))
        vz = int(np.floor(oz))

        # Step direction
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        step_z = 1 if dz > 0 else -1

        # t_delta: how much t to cross one voxel in each axis
        t_delta_x = abs(1.0 / dx) if dx != 0 else 1e30
        t_delta_y = abs(1.0 / dy) if dy != 0 else 1e30
        t_delta_z = abs(1.0 / dz) if dz != 0 else 1e30

        # t_max: t at which the ray crosses the first boundary in each axis
        if dx > 0:
            t_max_x = (vx + 1 - ox) / dx
        elif dx < 0:
            t_max_x = (vx - ox) / dx
        else:
            t_max_x = 1e30

        if dy > 0:
            t_max_y = (vy + 1 - oy) / dy
        elif dy < 0:
            t_max_y = (vy - oy) / dy
        else:
            t_max_y = 1e30

        if dz > 0:
            t_max_z = (vz + 1 - oz) / dz
        elif dz < 0:
            t_max_z = (vz - oz) / dz
        else:
            t_max_z = 1e30

        dist = 0.0

        while dist < self.max_dist:
            # Step along the axis with smallest t_max
            if t_max_x < t_max_y and t_max_x < t_max_z:
                dist = t_max_x
                t_max_x += t_delta_x
                vx += step_x
                face = 0  # hit an x-face
            elif t_max_y < t_max_z:
                dist = t_max_y
                t_max_y += t_delta_y
                vy += step_y
                face = 1  # hit a y-face
            else:
                dist = t_max_z
                t_max_z += t_delta_z
                vz += step_z
                face = 2  # hit a z-face

            # Check bounds
            if vx < 0 or vx >= world.width or vy < 0 or vy >= world.depth or vz < 0 or vz >= world.height:
                return None, False

            block = world.get_block(vx, vy, vz)
            if block != BlockType.AIR:
                base_color = np.array(BLOCK_COLORS[block], dtype=np.float32)

                # Distance fog
                shade = 1.0 / (1.0 + 0.08 * dist * dist)

                # Darken side faces slightly for depth cues
                if face == 0:
                    shade *= 0.8
                elif face == 1:
                    shade *= 0.9

                color = np.clip(base_color * shade, 0, 255).astype(np.uint8)
                return color, True

        return None, False