import torch
import numpy as np

# Utility functions
def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.2):
    ''' 
    Calculate rays' intersection time (near and far) with aabb using PyTorch operations
    Args:
        rays_o: float, [N, 3]
        rays_d: float, [N, 3]
        aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
        min_near: float, scalar
    Returns:
        nears: float, [N]
        fars: float, [N]
    '''
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)
    N = rays_o.shape[0] # num rays

    nears = torch.full((N,), float('inf'), dtype=rays_o.dtype, device=rays_o.device)
    fars = torch.full((N,), float('inf'), dtype=rays_o.dtype, device=rays_o.device)

    for i in range(N):
        ox, oy, oz = rays_o[i]
        dx, dy, dz = rays_d[i]
        rdx, rdy, rdz = 1 / dx, 1 / dy, 1 / dz
        tmin = ((aabb[:3] - rays_o[i]) * torch.tensor([rdx, rdy, rdz])).max()
        tmax = ((aabb[3:] - rays_o[i]) * torch.tensor([rdx, rdy, rdz])).min()
        if tmax >= max(tmin, min_near):
            nears[i] = tmin
            fars[i] = tmax
    return nears, fars

def sph_from_ray(rays_o, rays_d, radius):
    ''' 
    get spherical coordinate on the background sphere from rays.
    Assume rays_o are inside the Sphere(radius).
    Args:
        rays_o: [N, 3]
        rays_d: [N, 3]
        radius: scalar, float
    Return:
        coords: [N, 2], in [-1, 1], theta and phi on a sphere. (further-surface)
    '''
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)
    N = rays_o.shape[0] # num rays

    coords = torch.empty(N, 2, dtype=rays_o.dtype, device=rays_o.device)

    for i in range(N):
        ox, oy, oz = rays_o[i]
        dx, dy, dz = rays_d[i]
        A = dx**2 + dy**2 + dz**2
        B = 2 * (ox * dx + oy * dy + oz * dz)
        C = ox**2 + oy**2 + oz**2 - radius**2
        t = (-B + (B**2 - 4 * A * C).sqrt()) / (2 * A)
        x, y, z = ox + t * dx, oy + t * dy, oz + t * dz
        theta = torch.atan2((x**2 + z**2).sqrt(), y)
        phi = torch.atan2(z, x)
        coords[i, 0] = 2 * theta / np.pi - 1
        coords[i, 1] = phi / np.pi
    return coords

def morton3D(coords):
    ''' 
    Args:
        coords: [N, 3], int32, in [0, 128)
    Returns:
        indices: [N], int32, in [0, 128^3)
    '''
    coords = coords.int()
    N = coords.shape[0]

    indices = torch.zeros(N, dtype=torch.int32, device=coords.device)
    for i in range(N):
        x, y, z = coords[i]
        indices[i] = morton3D_single(x, y, z)
    return indices

def morton3D_single(x, y, z):
    def expand_bits(v):
        v = (v * 0x00010001) & 0xFF0000FF
        v = (v * 0x00000101) & 0x0F00F00F
        v = (v * 0x00000011) & 0xC30C30C3
        v = (v * 0x00000005) & 0x49249249
        return v
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)

def morton3D_invert(indices):
    ''' 
    Args:
        indices: [N], int32, in [0, 128^3)
    Returns:
        coords: [N, 3], int32, in [0, 128)
    '''
    N = indices.shape[0]
    coords = torch.zeros(N, 3, dtype=torch.int32, device=indices.device)
    for i in range(N):
        coords[i] = morton3D_invert_single(indices[i])
    return coords

def morton3D_invert_single(x):
    def compact_bits(v):
        v = v & 0x49249249
        v = (v | (v >> 2)) & 0xc30c30c3
        v = (v | (v >> 4)) & 0x0f00f00f
        v = (v | (v >> 8)) & 0xff0000ff
        v = (v | (v >> 16)) & 0x0000ffff
        return v
    return torch.tensor([compact_bits(x >> i) for i in (0, 1, 2)], device=x.device)

def packbits(grid, thresh):
    ''' 
    Pack up the density grid into a bit field to accelerate ray marching.
    Args:
        grid: float, [C, H * H * H], assume H % 2 == 0
        thresh: float, threshold
    Returns:
        bitfield: uint8, [C, H * H * H / 8]
    '''
    C, H3 = grid.shape
    N = C * H3 // 8
    bitfield = torch.zeros(N, dtype=torch.uint8, device=grid.device)
    for i in range(N):
        bits = 0
        for j in range(8):
            bits |= (grid.view(-1)[i * 8 + j] > thresh).byte() << j
        bitfield[i] = bits
    return bitfield

# Ray marching functions for training and inference
def march_rays_train(rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False, dt_gamma=0, max_steps=1024):
    ''' 
    march rays to generate points (forward only)
    Args:
        rays_o/d: float, [N, 3]
        bound: float, scalar
        density_bitfield: uint8: [CHHH // 8]
        C: int
        H: int
        nears/fars: float, [N]
        step_counter: int32, (2), used to count the actual number of generated points.
        mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
        perturb: bool
        align: int, pad output so its size is dividable by align, set to -1 to disable.
        force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
        dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
        max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
    Returns:
        xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
        dirs: float, [M, 3], all generated points' view dirs.
        deltas: float, [M, 2], all generated points' deltas. (first for RGB, second for Depth)
        rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 2]] --> points belonging to rays[i, 0]
    '''
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)
    density_bitfield = density_bitfield.contiguous()
    N = rays_o.shape[0]
    M = N * max_steps
    if not force_all_rays and mean_count > 0:
        if align > 0:
            mean_count += align - mean_count % align
        M = mean_count
    xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
    dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
    deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
    rays = torch.zeros(N, 3, dtype=torch.int32, device=rays_o.device)
    if step_counter is None:
        step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device)
    if perturb:
        noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
    else:
        noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)
    # Marching loop
    for i in range(N):
        t = nears[i]
        xyz_idx = step_counter[0].item()
        ray_idx = step_counter[1].item()
        rays[ray_idx, 0] = i
        rays[ray_idx, 1] = xyz_idx
        step_counter[1] += 1
        last_t = t
        num_steps = 0
        while t < fars[i] and num_steps < max_steps:
            x = rays_o[i, 0] + t * rays_d[i, 0]
            y = rays_o[i, 1] + t * rays_d[i, 1]
            z = rays_o[i, 2] + t * rays_d[i, 2]
            dt = t * dt_gamma
            level = min(C - 1, max(0, int(torch.log2(dt * H / (2 * 1.7320508075688772)))))
            mip_bound = min(1.0 * 2**level, bound)
            nx = int((x * mip_bound + 1) * 0.5 * H)
            ny = int((y * mip_bound + 1) * 0.5 * H)
            nz = int((z * mip_bound + 1) * 0.5 * H)
            index = level * H**3 + morton3D_single(nx, ny, nz)
            if density_bitfield[index // 8] & (1 << (index % 8)):
                xyzs[xyz_idx, 0] = x
                xyzs[xyz_idx, 1] = y
                xyzs[xyz_idx, 2] = z
                dirs[xyz_idx, 0] = rays_d[i, 0]
                dirs[xyz_idx, 1] = rays_d[i, 1]
                dirs[xyz_idx, 2] = rays_d[i, 2]
                deltas[xyz_idx, 0] = dt
                deltas[xyz_idx, 1] = t - last_t
                last_t = t
                t += dt
                xyz_idx += 1
                step_counter[0] += 1
                num_steps += 1
            else:
                tx = ((nx + 0.5 + 0.5 * torch.sign(rays_d[i, 0])) * mip_bound - x) / rays_d[i, 0]
                ty = ((ny + 0.5 + 0.5 * torch.sign(rays_d[i, 1])) * mip_bound - y) / rays_d[i, 1]
                tz = ((nz + 0.5 + 0.5 * torch.sign(rays_d[i, 2])) * mip_bound - z) / rays_d[i, 2]
                t = min(tx, min(ty, tz))
    return xyzs, dirs, deltas, rays

def composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh=1e-4):
    ''' 
    composite rays' rgbs, according to the ray marching formula.
    Args:
        rgbs: float, [M, 3]
        sigmas: float, [M,]
        deltas: float, [M, 2]
        rays: int32, [N, 3]
    Returns:
        weights_sum: float, [N,], the alpha channel
        depth: float, [N, ], the Depth
        image: float, [N, 3], the RGB channel (after multiplying alpha!)
    '''
    M = sigmas.shape[0]
    N = rays.shape[0]
    weights_sum = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
    depth = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
    image = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
    for i in range(N):
        idx = rays[i, 0]
        offset = rays[i, 1]
        num_steps = rays[i, 2]
        if num_steps == 0:
            continue
        T = 1.0
        r, g, b, d, ws = 0, 0, 0, 0, 0
        for j in range(num_steps):
            alpha = 1.0 - torch.exp(-sigmas[offset + j] * deltas[offset + j, 0])
            weight = alpha * T
            r += weight * rgbs[offset + j, 0]
            g += weight * rgbs[offset + j, 1]
            b += weight * rgbs[offset + j, 2]
            d += weight * (deltas[offset + j, 1] + deltas[offset + j, 0])
            ws += weight
            T *= (1.0 - alpha)
            if T < T_thresh:
                break
        weights_sum[i] = ws
        depth[i] = d
        image[i, 0] = r
        image[i, 1] = g
        image[i, 2] = b
    return weights_sum, depth, image



def composite_rays_train_localization_int(sigmas, rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, rays, T_thresh=1e-4):
    ''' 
    composite rays' rgbs with localization and style, according to the ray marching formula.
    Args:
        rgbs: float, [M, 3]
        sigmas: float, [M,]
        loc_prob: float, [M]
        style_rgbs: float, [M, 3]
        beck_rgbs: float, [M, 3]
        deltas: float, [M, 2]
        rays: int32, [N, 3]
    Returns:
        weights_sum: float, [N,], the alpha channel
        depth: float, [N, ], the Depth
        image_origin: float, [N, 3], the original RGB channel
        image_prob: float, [N,], the localization probability
        image_style_origin: float, [N, 3], the style RGB channel
        image_back_origin: float, [N, 3], the background RGB channel
        image_localization_mix: float, [N, 3], the localization mix RGB channel
        image_style_mix: float, [N, 3], the style mix RGB channel
        image_back_mix: float, [N, 3], the background mix RGB channel
    '''
    M = sigmas.shape[0]
    N = rays.shape[0]
    weights_sum = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
    depth = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
    image_origin = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
    image_prob = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
    image_style_origin = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
    image_back_origin = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
    image_localization_mix = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
    image_style_mix = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
    image_back_mix = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)

    r_yellow, g_yellow, b_yellow = 0.8, 1.0, 0.0
    r_gray, g_gray, b_gray = 0.71, 0.71, 0.71

    for i in range(N):
        idx = rays[i, 0]
        offset = rays[i, 1]
        num_steps = rays[i, 2]
        if num_steps == 0:
            continue
        T = 1.0
        r, g, b, d, ws = 0, 0, 0, 0, 0
        localization_prob = 0
        r_stylization, g_stylization, b_stylization = 0, 0, 0
        r_background, g_background, b_background = 0, 0, 0

        for j in range(num_steps):
            alpha = 1.0 - torch.exp(-sigmas[offset + j] * deltas[offset + j, 0])
            weight = alpha * T
            localization_prob += weight * loc_prob[offset + j]

            r += weight * rgbs[offset + j, 0]
            g += weight * rgbs[offset + j, 1]
            b += weight * rgbs[offset + j, 2]

            r_stylization += weight * style_rgbs[offset + j, 0]
            g_stylization += weight * style_rgbs[offset + j, 1]
            b_stylization += weight * style_rgbs[offset + j, 2]

            r_background += weight * beck_rgbs[offset + j, 0]
            g_background += weight * beck_rgbs[offset + j, 1]
            b_background += weight * beck_rgbs[offset + j, 2]

            d += weight * (deltas[offset + j, 1] + deltas[offset + j, 0])
            ws += weight
            T *= (1.0 - alpha)
            if T < T_thresh:
                break

        weights_sum[i] = ws
        depth[i] = d
        image_origin[i, 0] = r
        image_origin[i, 1] = g
        image_origin[i, 2] = b
        image_prob[i] = localization_prob
        image_style_origin[i, 0] = r_stylization
        image_style_origin[i, 1] = g_stylization
        image_style_origin[i, 2] = b_stylization
        image_back_origin[i, 0] = r_background
        image_back_origin[i, 1] = g_background
        image_back_origin[i, 2] = b_background

        image_localization_mix[i, 0] = localization_prob * r_yellow + (1 - localization_prob) * r_gray
        image_localization_mix[i, 1] = localization_prob * g_yellow + (1 - localization_prob) * g_gray
        image_localization_mix[i, 2] = localization_prob * b_yellow + (1 - localization_prob) * b_gray

        image_style_mix[i, 0] = localization_prob * r_stylization + (1 - localization_prob) * r_gray
        image_style_mix[i, 1] = localization_prob * g_stylization + (1 - localization_prob) * g_gray
        image_style_mix[i, 2] = localization_prob * b_stylization + (1 - localization_prob) * b_gray

        image_back_mix[i, 0] = localization_prob * r_yellow + (1 - localization_prob) * r_background
        image_back_mix[i, 1] = localization_prob * g_yellow + (1 - localization_prob) * g_background
        image_back_mix[i, 2] = localization_prob * b_yellow + (1 - localization_prob) * b_background

    return weights_sum, depth, image_origin, image_prob, image_style_origin, image_back_origin, image_localization_mix, image_style_mix, image_back_mix

def composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh=1e-2):
    ''' 
    composite rays' rgbs, according to the ray marching formula. (for inference)
    Args:
        n_alive: int, number of alive rays
        n_step: int, how many steps we march
        rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
        rays_t: float, [N], the alive rays' time
        sigmas: float, [n_alive * n_step,]
        rgbs: float, [n_alive * n_step, 3]
        deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
    In-place Outputs:
        weights_sum: float, [N,], the alpha channel
        depth: float, [N,], the depth value
        image: float, [N, 3], the RGB channel (after multiplying alpha!)
    '''
    for i in range(n_alive):
        idx = rays_alive[i]
        t = rays_t[idx]
        T = 1.0
        r, g, b, d = 0, 0, 0, 0
        for j in range(n_step):
            if deltas[i * n_step + j, 0] == 0:
                break
            alpha = 1.0 - torch.exp(-sigmas[i * n_step + j] * deltas[i * n_step + j, 0])
            weight = alpha * T
            r += weight * rgbs[i * n_step + j, 0]
            g += weight * rgbs[i * n_step + j, 1]
            b += weight * rgbs[i * n_step + j, 2]
            t += deltas[i * n_step + j, 1]
            d += weight * t
            T *= (1.0 - alpha)
            if T < T_thresh:
                break
        if j < n_step:
            rays_alive[i] = -1
        else:
            rays_t[idx] = t
        weights_sum[idx] = 1 - T
        depth[idx] = d
        image[idx, 0] = r
        image[idx, 1] = g
        image[idx, 2] = b

def composite_rays_localization_int(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, weights_sum, depth, image_origin, image_prob, image_style_origin, image_back_origin, image_localization_mix, image_style_mix, image_back_mix, T_thresh=1e-2):
    ''' 
    composite rays' rgbs with localization and style, according to the ray marching formula. (for inference)
    Args:
        n_alive: int, number of alive rays
        n_step: int, how many steps we march
        rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
        rays_t: float, [N], the alive rays' time
        sigmas: float, [n_alive * n_step,]
        rgbs: float, [n_alive * n_step, 3]
        loc_prob: float, [n_alive * n_step]
        style_rgbs: float, [n_alive * n_step, 3]
        beck_rgbs: float, [n_alive * n_step, 3]
        deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
    In-place Outputs:
        weights_sum: float, [N,], the alpha channel
        depth: float, [N,], the depth value
        image_origin: float, [N, 3], the original RGB channel
        image_prob: float, [N,], the localization probability
        image_style_origin: float, [N, 3], the style RGB channel
        image_back_origin: float, [N, 3], the background RGB channel
        image_localization_mix: float, [N, 3], the localization mix RGB channel
        image_style_mix: float, [N, 3], the style mix RGB channel
        image_back_mix: float, [N, 3], the background mix RGB channel
    '''
    r_yellow, g_yellow, b_yellow = 0.8, 1.0, 0.0
    r_gray, g_gray, b_gray = 0.71, 0.71, 0.71

    for i in range(n_alive):
        idx = rays_alive[i]
        t = rays_t[idx]
        T = 1.0
        r, g, b, d = 0, 0, 0, 0
        localization_prob = 0
        r_stylization, g_stylization, b_stylization = 0, 0, 0
        r_background, g_background, b_background = 0, 0, 0

        for j in range(n_step):
            if deltas[i * n_step + j, 0] == 0:
                break
            alpha = 1.0 - torch.exp(-sigmas[i * n_step + j] * deltas[i * n_step + j, 0])
            weight = alpha * T
            r += weight * rgbs[i * n_step + j, 0]
            g += weight * rgbs[i * n_step + j, 1]
            b += weight * rgbs[i * n_step + j, 2]
            localization_prob += weight * loc_prob[i * n_step + j]
            r_stylization += weight * style_rgbs[i * n_step + j, 0]
            g_stylization += weight * style_rgbs[i * n_step + j, 1]
            b_stylization += weight * style_rgbs[i * n_step + j, 2]
            r_background += weight * beck_rgbs[i * n_step + j, 0]
            g_background += weight * beck_rgbs[i * n_step + j, 1]
            b_background += weight * beck_rgbs[i * n_step + j, 2]
            t += deltas[i * n_step + j, 1]
            d += weight * t
            T *= (1.0 - alpha)
            if T < T_thresh:
                break
        if j < n_step:
            rays_alive[i] = -1
        else:
            rays_t[idx] = t
        weights_sum[idx] = 1 - T
        depth[idx] = d
        image_origin[idx, 0] = r
        image_origin[idx, 1] = g
        image_origin[idx, 2] = b
        image_prob[idx] = localization_prob
        image_style_origin[idx, 0] = r_stylization
        image_style_origin[idx, 1] = g_stylization
        image_style_origin[idx, 2] = b_stylization
        image_back_origin[idx, 0] = r_background
        image_back_origin[idx, 1] = g_background
        image_back_origin[idx, 2] = b_background

        image_localization_mix[idx, 0] = localization_prob * r_yellow + (1 - localization_prob) * r_gray
        image_localization_mix[idx, 1] = localization_prob * g_yellow + (1 - localization_prob) * g_gray
        image_localization_mix[idx, 2] = localization_prob * b_yellow + (1 - localization_prob) * b_gray

        image_style_mix[idx, 0] = localization_prob * r_stylization + (1 - localization_prob) * r_gray
        image_style_mix[idx, 1] = localization_prob * g_stylization + (1 - localization_prob) * g_gray
        image_style_mix[idx, 2] = localization_prob * b_stylization + (1 - localization_prob) * b_gray

        image_back_mix[idx, 0] = localization_prob * r_yellow + (1 - localization_prob) * r_background
        image_back_mix[idx, 1] = localization_prob * g_yellow + (1 - localization_prob) * g_background
        image_back_mix[idx, 2] = localization_prob * b_yellow + (1 - localization_prob) * b_background
