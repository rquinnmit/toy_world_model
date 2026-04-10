import numpy as np
from env.voxel_world import BLOCK_COLORS, BlockType

# Precompute a color lookup table: block_type_id -> (R, G, B)
_COLOR_LUT = np.zeros((max(int(b) for b in BlockType) + 1, 3), dtype=np.float32)
for bt, color in BLOCK_COLORS.items():
    _COLOR_LUT[int(bt)] = color


class Renderer:
    def __init__(self, width=64, height=64, fov=np.pi / 3, max_dist=30.0, max_steps=128):
        self.width = width
        self.height = height
        self.fov = fov
        self.max_dist = max_dist
        self.max_steps = max_steps

    def render(self, world):
        eye_x, eye_y, eye_z = world.get_agent_eye_pos()
        yaw = world.agent_yaw

        # --- Build all ray directions at once ---
        px = np.arange(self.width, dtype=np.float64)
        py = np.arange(self.height, dtype=np.float64)
        px_grid, py_grid = np.meshgrid(px, py)

        angle_h = yaw + self.fov * (0.5 - px_grid / self.width)
        angle_v = self.fov * (0.5 - py_grid / self.height)

        ray_dx = np.cos(angle_h) * np.cos(angle_v)
        ray_dy = np.sin(angle_h) * np.cos(angle_v)
        ray_dz = -np.sin(angle_v)

        flat_dx = ray_dx.ravel()
        flat_dy = ray_dy.ravel()
        flat_dz = ray_dz.ravel()
        N = len(flat_dx)

        # --- DDA setup for all rays in parallel ---
        ox = np.full(N, eye_x)
        oy = np.full(N, eye_y)
        oz = np.full(N, eye_z)

        vx = np.floor(ox).astype(np.int32)
        vy = np.floor(oy).astype(np.int32)
        vz = np.floor(oz).astype(np.int32)

        step_x = np.where(flat_dx > 0, 1, -1).astype(np.int32)
        step_y = np.where(flat_dy > 0, 1, -1).astype(np.int32)
        step_z = np.where(flat_dz > 0, 1, -1).astype(np.int32)

        with np.errstate(divide="ignore"):
            inv_dx = np.where(flat_dx != 0, 1.0 / flat_dx, 1e30)
            inv_dy = np.where(flat_dy != 0, 1.0 / flat_dy, 1e30)
            inv_dz = np.where(flat_dz != 0, 1.0 / flat_dz, 1e30)

        t_delta_x = np.abs(inv_dx)
        t_delta_y = np.abs(inv_dy)
        t_delta_z = np.abs(inv_dz)

        t_max_x = np.where(flat_dx > 0, (vx + 1 - ox) * inv_dx,
                           np.where(flat_dx < 0, (vx - ox) * inv_dx, 1e30))
        t_max_y = np.where(flat_dy > 0, (vy + 1 - oy) * inv_dy,
                           np.where(flat_dy < 0, (vy - oy) * inv_dy, 1e30))
        t_max_z = np.where(flat_dz > 0, (vz + 1 - oz) * inv_dz,
                           np.where(flat_dz < 0, (vz - oz) * inv_dz, 1e30))

        # --- March all rays simultaneously ---
        hit = np.zeros(N, dtype=bool)
        hit_dist = np.full(N, self.max_dist)
        hit_block = np.zeros(N, dtype=np.int32)
        hit_face = np.zeros(N, dtype=np.int32)

        active = np.ones(N, dtype=bool)

        for _ in range(self.max_steps):
            if not np.any(active):
                break

            # Determine which axis each active ray steps along
            step_x_mask = active & (t_max_x < t_max_y) & (t_max_x < t_max_z)
            step_y_mask = active & ~step_x_mask & (t_max_y < t_max_z)
            step_z_mask = active & ~step_x_mask & ~step_y_mask

            # Record distances before stepping
            dist = np.where(step_x_mask, t_max_x,
                            np.where(step_y_mask, t_max_y, t_max_z))

            # Deactivate rays that exceeded max distance
            too_far = active & (dist >= self.max_dist)
            active[too_far] = False

            # Step voxel coordinates
            vx[step_x_mask] += step_x[step_x_mask]
            vy[step_y_mask] += step_y[step_y_mask]
            vz[step_z_mask] += step_z[step_z_mask]

            # Advance t_max
            t_max_x[step_x_mask] += t_delta_x[step_x_mask]
            t_max_y[step_y_mask] += t_delta_y[step_y_mask]
            t_max_z[step_z_mask] += t_delta_z[step_z_mask]

            # Deactivate rays that left the grid
            out_of_bounds = active & (
                (vx < 0) | (vx >= world.width) |
                (vy < 0) | (vy >= world.depth) |
                (vz < 0) | (vz >= world.height)
            )
            active[out_of_bounds] = False

            # Look up block types for active, in-bounds rays
            check = active & ~out_of_bounds
            if not np.any(check):
                continue

            check_idx = np.where(check)[0]
            blocks = world.grid[vx[check_idx], vy[check_idx], vz[check_idx]]

            solid = blocks != int(BlockType.AIR)
            solid_idx = check_idx[solid]

            if len(solid_idx) > 0:
                hit[solid_idx] = True
                hit_dist[solid_idx] = dist[solid_idx]
                hit_block[solid_idx] = blocks[solid]
                hit_face[solid_idx] = np.where(
                    step_x_mask[solid_idx], 0,
                    np.where(step_y_mask[solid_idx], 1, 2)
                )
                active[solid_idx] = False

        # --- Shade and assemble the final image ---
        sky = np.array(BLOCK_COLORS[BlockType.AIR], dtype=np.float32)
        image = np.tile(sky, (N, 1))

        if np.any(hit):
            hit_idx = np.where(hit)[0]
            base_colors = _COLOR_LUT[hit_block[hit_idx]]

            shade = 1.0 / (1.0 + 0.08 * hit_dist[hit_idx] ** 2)
            face_mul = np.where(hit_face[hit_idx] == 0, 0.8,
                                np.where(hit_face[hit_idx] == 1, 0.9, 1.0))
            shade *= face_mul

            image[hit_idx] = base_colors * shade[:, np.newaxis]

        image = np.clip(image, 0, 255).astype(np.uint8)
        return image.reshape(self.height, self.width, 3)
