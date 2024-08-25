import math
import torch
import torch.nn as nn
from loguru import logger

from src.latent_nerf.configs.render_config import RenderConfig
from .render_utils import sample_pdf, custom_meshgrid, safe_normalize


class NeRFRenderer(nn.Module):
    def __init__(self, cfg:RenderConfig, latent_mode:bool=True):
        super().__init__()

        self.opt = cfg
        self.bound = cfg.bound
        self.cascade = 1 + math.ceil(math.log2(cfg.bound))
        self.grid_size = 128
        self.cuda_ray = cfg.cuda_ray
        self.min_near = cfg.min_near
        self.density_thresh = cfg.density_thresh
        self.bg_radius = cfg.bg_radius
        self.latent_mode = latent_mode
        self.img_dims = 3+1 if self.latent_mode else 3
        self.first_arc = cfg.first_arc
        self.second_arc = cfg.second_arc
        self.third_arc = cfg.third_arc
        if self.cuda_ray:
            logger.info('Loading CUDA ray marching module (compiling might take a while)...')
            from src.latent_nerf.raymarching import raymarchingrgb, raymarchinglatent
            logger.info('\tDone.')
            self.raymarching = raymarchinglatent if self.latent_mode else raymarchingrgb
        elif cfg.train_localization:
            logger.info('Loading CUDA ray marching module (compiling might take a while)...')
            from src.latent_nerf.raymarching import raymarchingrgb, raymarchinglatent
            logger.info('\tDone.')
            self.raymarching = raymarchinglatent if self.latent_mode else raymarchingrgb

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-cfg.bound, -cfg.bound, -cfg.bound, cfg.bound, cfg.bound, cfg.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    
    def forward(self, x, d):
        raise NotImplementedError()

    def forward_localization(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def density_localization(self,x):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0
    def run_localization(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        # query SDF and RGB
        density_outputs = self.density_localization(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density_localization(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]

        alphas_prob = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1) * density_outputs['loc_prob'].squeeze(-1))

        alphas_prob_2 = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1) * (1- density_outputs['loc_prob'].squeeze(-1)))

        alphas_prob_3 = 1 - torch.exp(-deltas * density_outputs['loc_prob'].squeeze(-1))

        alphas_prob_4 = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) * density_outputs['loc_prob'].squeeze(-1)

        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]

        alphas_shifted_prob = torch.cat([torch.ones_like(alphas_prob[..., :1]), 1 - alphas_prob + 1e-15], dim=-1)

        alphas_shifted_prob_2 = torch.cat([torch.ones_like(alphas_prob_2[..., :1]), 1 - alphas_prob_2 + 1e-15], dim=-1)

        alphas_shifted_prob_3 = torch.cat([torch.ones_like(alphas_prob_3[..., :1]), 1 - alphas_prob_3 + 1e-15], dim=-1)

        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        weights_prob = alphas_prob * torch.cumprod(alphas_shifted_prob, dim=-1)[..., :-1]

        weights_prob_2 = alphas_prob_2 * torch.cumprod(alphas_shifted_prob_2, dim=-1)[..., :-1]

        weights_prob_3 = alphas_prob_3 * torch.cumprod(alphas_shifted_prob_3, dim=-1)[..., :-1]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        sigma, rgbs, normals , loc_prob, style_rgbs, back_rgbs = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, self.img_dims) # [N, T+t, 3]
        loc_prob = loc_prob.view(N, -1, 1)
        style_rgbs = style_rgbs.view(N, -1, self.img_dims)
        back_rgbs = back_rgbs.view(N, -1, self.img_dims)

        # orientation loss
        if normals is not None:
            print('Normals not None!')
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.mean()

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]

        weights_sum_prob = weights_prob.sum(dim=-1) + weights_prob_2.sum(dim=-1)
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]
        image_prob = torch.sum(weights.unsqueeze(-1) * loc_prob, dim=-2)
        image_style_origin = torch.sum(weights.unsqueeze(-1) * style_rgbs, dim=-2)
        image_back_origin = torch.sum(weights.unsqueeze(-1) * back_rgbs, dim=-2)





        
        r_yellow, g_yellow, b_yellow = 0.8, 1.0, 0.0
        r_gray, g_gray, b_gray = 0.71, 0.71, 0.71

        yellow_tensor = torch.tensor([r_yellow, g_yellow, b_yellow], device=device).view(1, 1, 3)
        gray_tensor = torch.tensor([r_gray, g_gray, b_gray], device=device).view(1, 1, 3)


        image_loc_test = torch.sum(weights.unsqueeze(-1) * ((loc_prob * yellow_tensor) + (1-loc_prob) * gray_tensor), dim=-2)
        image_style_test = torch.sum(weights.unsqueeze(-1) * ((loc_prob * style_rgbs) + (1-loc_prob) * gray_tensor), dim=-2)
        image_beck_test = torch.sum(weights.unsqueeze(-1) * ((loc_prob * yellow_tensor) + (1-loc_prob) * back_rgbs), dim=-2)


        image_loc_test2 = torch.sum(weights_prob.unsqueeze(-1) * yellow_tensor + weights_prob_2.unsqueeze(-1) * gray_tensor, dim=-2)
        image_style_test2 = torch.sum(weights_prob.unsqueeze(-1) * style_rgbs + weights_prob_2.unsqueeze(-1) * gray_tensor, dim=-2)
        image_beck_test2 = torch.sum(weights_prob.unsqueeze(-1) * yellow_tensor + weights_prob_2.unsqueeze(-1) * back_rgbs, dim=-2)


        image_loc_test3 = torch.sum(weights.unsqueeze(-1) * (weights_prob_3.unsqueeze(-1) * yellow_tensor + (1 - weights_prob_3.unsqueeze(-1)) * gray_tensor), dim=-2)
        image_style_test3 = torch.sum(weights.unsqueeze(-1) * (weights_prob_3.unsqueeze(-1) * style_rgbs + (1 - weights_prob_3.unsqueeze(-1)) * gray_tensor), dim=-2)
        image_beck_test3 = torch.sum(weights.unsqueeze(-1) * (weights_prob_3.unsqueeze(-1) * yellow_tensor + (1 - weights_prob_3.unsqueeze(-1)) * back_rgbs), dim=-2)


        # image_localization_mix = image_prob.unsqueeze(-1) * torch.tensor([r_yellow, g_yellow, b_yellow], device=device) + (1 - image_prob.unsqueeze(-1)) * torch.tensor([r_gray, g_gray, b_gray], device=device)
        # image_style_mix = image_prob.unsqueeze(-1) * image_style_origin + (1 - image_prob.unsqueeze(-1)) * torch.tensor([r_gray, g_gray, b_gray], device=device)
        # image_back_mix = image_prob.unsqueeze(-1) * torch.tensor([r_yellow, g_yellow, b_yellow], device=device) + (1 - image_prob.unsqueeze(-1)) * image_back_origin
        
        # for gray
        image_style_mix = image_prob * image_style_origin + ((1 - image_prob) * gray_tensor)
        image_back_mix = image_prob * yellow_tensor + ((1 - image_prob) * image_back_origin)
        image_localization_mix = image_prob * yellow_tensor + ((1 - image_prob) * gray_tensor)

            

        # image_style_mix = image_prob * image_style_origin + ((1 - image_prob) * image)
        # image_back_mix = image_prob * yellow_tensor + ((1 - image_prob) * image_back_origin)
        # image_localization_mix = image_prob * yellow_tensor + ((1 - image_prob) * image)

        mixed_image = image_prob * image_style_origin + ((1 - image_prob) * image)


        # mix background color
        # if self.bg_radius > 0:
        #     # use the bg model to calculate bg_color
        #     # sph = self.raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
        #     bg_color = self.background(rays_d.reshape(-1, 3)) # [N, 3]
        # elif bg_color is None:
        #     bg_color = 1
        
        bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_style_origin = image_style_origin + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_back_origin = image_back_origin + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_localization_mix = image_localization_mix + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_style_mix = image_style_mix + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_back_mix = image_back_mix + (1 - weights_sum).unsqueeze(-1) * bg_color


        image_loc_test = image_loc_test + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_style_test = image_style_test + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_beck_test = image_beck_test + (1 - weights_sum).unsqueeze(-1) * bg_color


        image_loc_test2 = image_loc_test2 + (1 - weights_sum_prob).unsqueeze(-1) * bg_color
        image_style_test2 = image_style_test2 + (1 - weights_sum_prob).unsqueeze(-1) * bg_color
        image_beck_test2 = image_beck_test2 + (1 - weights_sum_prob).unsqueeze(-1) * bg_color


        image_loc_test3 = image_loc_test3 + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_style_test3 = image_style_test3 + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_beck_test3 = image_beck_test3 + (1 - weights_sum).unsqueeze(-1) * bg_color


        mixed_image = mixed_image + (1 - weights_sum).unsqueeze(-1) * bg_color


        image = image.view(*prefix, self.img_dims)
        image_prob = image_prob.view(*prefix, 1)
        image_style_origin = image_style_origin.view(*prefix, self.img_dims)
        image_back_origin = image_back_origin.view(*prefix, self.img_dims)
        image_localization_mix = image_localization_mix.view(*prefix, self.img_dims)
        image_style_mix = image_style_mix.view(*prefix, self.img_dims)
        image_back_mix = image_back_mix.view(*prefix, self.img_dims)



        # image_loc_test = image_loc_test.view(*prefix, self.img_dims)
        # image_style_test = image_style_test.view(*prefix, self.img_dims)
        # image_beck_test = image_beck_test.view(*prefix, self.img_dims)

        image_loc_test2 = image_loc_test2.view(*prefix, self.img_dims)
        image_style_test2 = image_style_test2.view(*prefix, self.img_dims)
        image_beck_test2 = image_beck_test2.view(*prefix, self.img_dims)

        image_loc_test3 = image_loc_test3.view(*prefix, self.img_dims)
        image_style_test3 = image_style_test3.view(*prefix, self.img_dims)
        image_beck_test3 = image_beck_test3.view(*prefix, self.img_dims)


        mixed_image = mixed_image.view(*prefix, self.img_dims)


        depth = depth.view(*prefix)

        mask = (nears < fars).reshape(*prefix)

        results['image_origin'] = image
        results['image_style_origin'] = image_style_origin
        results['image_back_origin'] = image_back_origin
        results['image_prob'] = image_prob
        results['image_localization_mix'] = image_localization_mix
        results['image_style_mix'] = image_style_mix
        results['image_back_mix'] = image_back_mix

        # for test
        if self.first_arc:
            results['image_localization_mix'] = image_loc_test
            results['image_style_mix'] = image_style_test
            results['image_back_mix'] = image_beck_test

        if self.second_arc:
            results['image_localization_mix'] = image_loc_test2
            results['image_style_mix'] = image_style_test2
            results['image_back_mix'] = image_beck_test2

        if self.third_arc:
            results['image_localization_mix'] = image_loc_test3
            results['image_style_mix'] = image_style_test3
            results['image_back_mix'] = image_beck_test3 

        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        results['mixed_image'] = mixed_image
        # results['sigmas'] = density_outputs['sigma'].view(N, num_steps)
        # results['alphas'] = alphas

        return results

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        sigmas, rgbs, normals, loc_prob, style_rgbs, beck_rgbs = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, self.img_dims) # [N, T+t, 3]

        # orientation loss
        if normals is not None:
            print('Normals not None!')
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.mean()

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            # sph = self.raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        results['sigmas'] = density_outputs['sigma'].view(N, num_steps)
        results['alphas'] = alphas

        return results

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, disable_background=False):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        results = {}
        xyzs, sigmas = None, None
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = self.raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            
            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)

            weights_sum, depth, image = self.raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)

            # orientation loss
            if normals is not None:
                weights = 1 - torch.exp(-sigmas)
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, self.img_dims, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = self.raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                self.raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

        # mix background color
        if self.bg_radius > 0 and not disable_background:
            
            # use the bg model to calculate bg_color
            # sph = self.raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(rays_d) # [N, 3]

        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, self.img_dims)

        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        results['sigmas'] = sigmas
        results['rgbs'] = rgbs

        return results



    def run_cuda_localization(self, rays_o, rays_d, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, disable_background=False):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        results = {}
        xyzs, sigmas = None, None
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = self.raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            
            sigmas, rgbs, normals, loc_prob, style_rgbs, beck_rgbs = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            
            # weights_sum, depth, image, image_style, image_back = self.raymarching.composite_rays_train_localization(sigmas, rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, rays, T_thresh)

            # weights_sum, depth, image_origin, image_prob, image_style_origin, image_back_origin, image_localization_mix, image_style_mix, image_back_mix = self.raymarching.composite_rays_train_localization_int(sigmas, rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, rays, T_thresh)
            weights_sum, depth, image_origin, image_prob, image_style_origin, image_back_origin, image_localization_mix, image_style_mix, image_back_mix = self.raymarching.composite_rays_train_localization_int(sigmas, rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, rays, T_thresh)
            # image = image_localization_mix
            # image_style = image_style_mix
            # image_back = image_back_mix

            # orientation loss
            if normals is not None:
                weights = 1 - torch.exp(-sigmas)
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()

        else:           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            # TO GO BACK TO THE ORIGIN ARCHTECTURE COMMENT IN THE FOLLOWING AND COMMENT OUT AFTERWARD 
            # DONT FORGET TO COMMENT OUT THE COMPOSITE_RAYS ASS WELL
            # image = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            # image_style = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            # image_back = torch.zeros(N, self.img_dims, dtype=dtype, device=device)


            image_origin = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            image_prob = torch.zeros(N, dtype=dtype, device=device)
            image_style_origin = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            image_back_origin = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            image_localization_mix = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            image_style_mix = torch.zeros(N, self.img_dims, dtype=dtype, device=device)
            image_back_mix = torch.zeros(N, self.img_dims, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break


                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = self.raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, normals, loc_prob, style_rgbs, beck_rgbs = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                # self.raymarching.composite_rays_localization(n_alive, n_step, rays_alive, rays_t, sigmas,
                #                                              rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, weights_sum,
                #                                              depth, image, image_style, image_back, T_thresh)
                self.raymarching.composite_rays_localization_int(n_alive, n_step, rays_alive, rays_t, sigmas,
                                                             rgbs, loc_prob, style_rgbs, beck_rgbs, deltas, weights_sum,
                                                             depth, image_origin, image_prob, image_style_origin,
                                                             image_back_origin, image_localization_mix,
                                                             image_style_mix, image_back_mix, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
        
        image = image_origin




        # mix background color
        if self.bg_radius > 0 and not disable_background:
            
            # use the bg model to calculate bg_color
            # sph = self.raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(rays_d) # [N, 3]

        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_style_origin = image_style_origin + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_back_origin = image_back_origin + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_localization_mix = image_localization_mix + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_style_mix = image_style_mix + (1 - weights_sum).unsqueeze(-1) * bg_color
        image_back_mix = image_back_mix + (1 - weights_sum).unsqueeze(-1) * bg_color


        image = image.view(*prefix, self.img_dims)
        image_prob = image_prob.view(*prefix, 1)
        image_style_origin = image_style_origin.view(*prefix, self.img_dims)
        image_back_origin = image_back_origin.view(*prefix, self.img_dims)
        image_localization_mix = image_localization_mix.view(*prefix, self.img_dims)
        image_style_mix = image_style_mix.view(*prefix, self.img_dims)
        image_back_mix = image_back_mix.view(*prefix, self.img_dims)

        depth = depth.view(*prefix)

        mask = (nears < fars).reshape(*prefix)

        results['image_origin'] = image
        results['image_style_origin'] = image_style_origin
        results['image_back_origin'] = image_back_origin
        results['image_prob'] = image_prob
        results['image_localization_mix'] = image_localization_mix
        results['image_style_mix'] = image_style_mix
        results['image_back_mix'] = image_back_mix
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        results['sigmas'] = sigmas
        results['rgbs'] = rgbs

        return results


    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = self.raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign 
                        tmp_grid[cas, indices] = sigmas
        
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = self.raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

    def render_localization(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda_localization    
        else:
            _run = self.run_localization #TODO edit

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, self.img_dims), device=device)
            image_prob = torch.empty((B, N, 1), device=device)
            image_style_origin = torch.empty((B, N, self.img_dims), device=device)
            image_back_origin = torch.empty((B, N, self.img_dims), device=device)
            image_localization_mix = torch.empty((B, N, self.img_dims), device=device)
            image_style_mix = torch.empty((B, N, self.img_dims), device=device)
            image_back_mix = torch.empty((B, N, self.img_dims), device=device)
            mixed_image = torch.empty((B, N, self.img_dims), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image_origin']
                    image_prob[b:b+1, head:tail] = results_['image_prob']
                    image_style_origin[b:b+1, head:tail] = results_['image_style_origin']
                    image_back_origin[b:b+1, head:tail] = results_['image_back_origin']
                    image_localization_mix[b:b+1, head:tail] = results_['image_localization_mix']
                    image_style_mix[b:b+1, head:tail] = results_['image_style_mix']
                    image_back_mix[b:b+1, head:tail] = results_['image_back_mix']
                    mixed_image[b:b+1, head:tail] = results_['mixed_image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image_origin'] = image
            results['image_prob'] = image_prob
            results['image_style_origin'] = image_style_origin
            results['image_back_origin'] = image_back_origin
            results['image_localization_mix'] = image_localization_mix
            results['image_style_mix'] = image_style_mix
            results['image_back_mix'] = image_back_mix
            results['weights_sum'] = weights_sum
            results['mixed_image'] = mixed_image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, self.img_dims), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results