import torch
from torch import nn

from src.latent_nerf.configs.render_config import RenderConfig
from .encoding import get_encoder
from .nerf_utils import trunc_exp, MLP, NeRFType, init_decoder_layer
from .render_utils import safe_normalize
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 cfg: RenderConfig,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 ):

        super().__init__(cfg, latent_mode=cfg.nerf_type == NeRFType.latent)


        self.train_localization = cfg.train_localization
        self.csd = cfg.csd
        self.third_arc = cfg.third_arc
        # self.localization_dim = cfg.localization_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        additional_dim_size = 1 if self.latent_mode else 0

        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, desired_resolution=2048 * self.bound)

        self.sigma_net = MLP(self.in_dim, 4 + additional_dim_size, hidden_dim, num_layers, bias=True)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg

            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)

            self.bg_net = MLP(self.in_dim_bg, 3 + additional_dim_size, hidden_dim_bg, num_layers_bg, bias=True)

        else:
            self.bg_net = None
    
        if self.train_localization:
            stylization_beckground_output_dim = 4 + additional_dim_size - 1
            # localization_dim = 1 if self.localization_dim else 1 + 2 * stylization_beckground_output_dim

            self.localization_net = MLP(self.in_dim, 1, hidden_dim, num_layers, bias=True)
            self.stylization_net = MLP(self.in_dim, stylization_beckground_output_dim, hidden_dim , num_layers, bias=True)
            self.background_net = MLP(self.in_dim, stylization_beckground_output_dim, hidden_dim, num_layers, bias=True)


        if cfg.nerf_type == NeRFType.latent_tune:
            self.decoder_layer = nn.Linear(4, 3, bias=False)
            init_decoder_layer(self.decoder_layer)
        else:
            self.decoder_layer = None

    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def common_forward_localization(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound)

        h_origin = self.sigma_net(h)        
        h_localization = self.localization_net(h)
        h_stylization = self.stylization_net(h)
        h_background = self.background_net(h)
        # TODO: WHO ARE HERE
        # if self.localization_dim:
        #     loc_prob = torch.sigmoid(h_localization[..., 0])
        loc_prob = torch.sigmoid(h_localization)
        style_rgbs = h_stylization
        back_rgbs = h_background
            
        sigma = trunc_exp(h_origin[..., 0] + self.gaussian(x))

        if self.third_arc:
            loc_prob = trunc_exp(h_localization[..., 0] + self.gaussian(x))

        albedo = h_origin[..., 1:]

        if self.decoder_layer is not None:
            albedo = self.decoder_layer(albedo)
            albedo = (albedo + 1) / 2
            style_rgbs = self.decoder_layer(style_rgbs)
            style_rgbs = (style_rgbs + 1) / 2
            back_rgbs = self.decoder_layer(back_rgbs)
            back_rgbs = (back_rgbs + 1) / 2
        elif not self.latent_mode:
            albedo = torch.sigmoid(h_origin[..., 1:])
            style_rgbs = torch.sigmoid(h_stylization)
            back_rgbs = torch.sigmoid(h_background)

        return sigma, albedo , loc_prob, style_rgbs, back_rgbs


    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound)

        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0] + self.gaussian(x))
        albedo = h[..., 1:]
        if self.decoder_layer is not None:
            albedo = self.decoder_layer(albedo)
            albedo = (albedo + 1) / 2
        elif not self.latent_mode:
            albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward(
            (x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward(
            (x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward(
            (x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward(
            (x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return normal

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            if self.train_localization:
                sigma, color , loc_prob, style_rgbs, back_rgbs = self.common_forward_localization(x)
                normal = None
            else:
                sigma, color = self.common_forward(x)
                normal = None

        else:
            # query normal
            if self.train_localization:
                sigma, albedo , loc_prob, style_rgbs, back_rgbs = self.common_forward_localization(x)
            else:
                sigma, albedo = self.common_forward(x)
            normal = self.finite_difference_normal(x)

            # normalize...
            normal = safe_normalize(normal)
            normal[torch.isnan(normal)] = 0

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ -l).clamp(min=0)  # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
                if self.train_localization:
                    style_rgbs = color
                    back_rgbs = color
            elif shading == 'normal':
                color = (normal + 1) / 2
                if self.train_localization:
                    style_rgbs = color
                    back_rgbs = color
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
                if self.train_localization:
                    style_rgbs = style_rgbs * lambertian.unsqueeze(-1)
                    back_rgbs = back_rgbs * lambertian.unsqueeze(-1)
            if self.latent_mode:
                # pad color with a single dimension of zeros
                color = torch.cat([color, torch.zeros((color.shape[0], 1), device=color.device)], axis=1)
                if self.train_localization:
                    style_rgbs = torch.cat([style_rgbs, torch.zeros((style_rgbs.shape[0], 1), device=style_rgbs.device)], axis=1)
                    back_rgbs = torch.cat([back_rgbs, torch.zeros((back_rgbs.shape[0], 1), device=back_rgbs.device)], axis=1)

        ret = (sigma, color, normal)
        if self.train_localization:
            ret = (sigma, color, normal , loc_prob, style_rgbs, back_rgbs)
        return ret

    def density_localization(self, x):
        # x: [N, 3], in [-bound, bound]
        # TODO- understand when should i use common_forward_localization
        sigma, albedo , loc_prob, style_rgbs, back_rgbs = self.common_forward_localization(x)

        return {
            'sigma': sigma,
            'albedo': albedo,
            'loc_prob': loc_prob,
            'style_rgbs': style_rgbs,
            'back_rgbs': back_rgbs
        }
    
    def density(self, x):
        
        sigma, albedo = self.common_forward(x)

        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def background(self, d):

        h = self.encoder_bg(d)  # [N, C]

        rgbs = self.bg_net(h)

        if self.decoder_layer is not None:
            rgbs = self.decoder_layer(rgbs)
            rgbs = (rgbs + 1) / 2
        elif not self.latent_mode:
            rgbs = torch.sigmoid(rgbs)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]
        if self.decoder_layer is not None:
            params.append({'params': self.decoder_layer.parameters(), 'lr': lr})

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        if self.train_localization:
            params = []
            params.append({'params': self.localization_net.parameters(), 'lr': lr})
            params.append({'params': self.stylization_net.parameters(), 'lr': lr})
            params.append({'params': self.background_net.parameters(), 'lr': lr})
        if self.csd:
            params = []
            params.append({'params': self.localization_net.parameters()})
            params.append({'params': self.stylization_net.parameters()})
            params.append({'params': self.background_net.parameters()})

        return params