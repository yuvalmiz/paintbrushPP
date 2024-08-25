import sys
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List

import imageio
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.models.renderer import NeRFRenderer
from src.latent_nerf.training.nerf_dataset import NeRFDataset
from src.stable_diffusion import StableDiffusion
from src.utils import make_path, tensor2numpy
from src.latent_nerf.guidance.csd import CSD



class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.nerf = self.init_nerf()
        self.diffusion = self.init_diffusion()
        if not self.cfg.render.csd:
            self.text_z = self.calc_text_embeddings()
        if not self.cfg.render.train_localization:
            self.losses = self.init_losses()
            self.optimizer, self.scaler = self.init_optimizer()
        if self.cfg.render.csd:
            self.diffusion = self.init_diffusion_csd()
            self.preprocess_text_csd()
            self.text_z, self.text_z_neg = self.calc_text_embeddings_csd(self.cfg.guide_localization.localization_prompt)
            self.style_text_z, self.style_text_z_neg = self.calc_text_embeddings_csd(self.cfg.guide_localization.style_prompt)
            self.background_text_z, self.background_text_z_neg = self.calc_text_embeddings_csd(self.cfg.guide_localization.background_prompt)
            del self.diffusion.pipe.text_encoder # delete text encoder to save memory
            torch.cuda.empty_cache()

        self.dataloaders = self.init_dataloaders()

        self.past_checkpoints = []
        if self.cfg.render.train_localization:
            self.load_checkpoint(self.cfg.render.nerf_path, model_only=True)
            for param in self.nerf.sigma_net.parameters():
                param.requires_grad = False
            for param in self.nerf.encoder.parameters():
                param.requires_grad = False
            if self.nerf.decoder_layer is not None:
                for param in self.nerf.decoder_layer.parameters():
                    param.requires_grad = False
            self.optimizer, self.scaler = self.init_optimizer()
            if self.cfg.render.csd:
                self.optim = self.init_optimizer_csd()
        else:
            if self.cfg.optim.resume:
                self.load_checkpoint(model_only=False)
            if self.cfg.optim.ckpt is not None:
                self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_nerf(self) -> NeRFRenderer:
        if self.cfg.render.backbone == 'grid':
            from src.latent_nerf.models.network_grid import NeRFNetwork
        else:
            raise ValueError(f'{self.cfg.render.backbone} is not a valid backbone name')

        model = NeRFNetwork(self.cfg.render).to(self.device)
        logger.info(
            f'Loaded {self.cfg.render.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> StableDiffusion:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          latent_mode=self.nerf.latent_mode)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model


    def init_diffusion_csd(self):
        diffusion = CSD()
        if self.cfg.guide_localization.cascaded:
            self.diffusion_II = CSD(stage=2)
            del self.diffusion_II.pipe.text_encoder
            torch.cuda.empty_cache()

        return diffusion

    def preprocess_text_csd(self):
        if self.cfg.guide_localization.prefix != "":
            self.cfg.guide_localization.prefix += " "
        if self.cfg.guide_localization.localization_prompt is None:
            self.cfg.guide_localization.localization_prompt = f"a 3d render of a gray {self.cfg.guide_localization.object_name} with {self.cfg.guide_localization.prefix}yellow {self.cfg.guide_localization.edit}"
        if self.cfg.guide_localization.style_prompt is None:
            if self.cfg.guide_localization.style == '':
                self.cfg.guide_localization.style_prompt = f"a 3d render of a gray {self.cfg.guide_localization.object_name} with {self.cfg.guide_localization.prefix}{self.cfg.guide_localization.edit}"    
            else:
                self.cfg.guide_localization.style_prompt = f"a 3d render of a gray {self.cfg.guide_localization.object_name} with {self.cfg.guide_localization.prefix}{self.cfg.guide_localization.style} {self.cfg.guide_localization.edit}"
        if self.cfg.guide_localization.background_prompt is None:
            self.cfg.guide_localization.background_prompt = f"a 3d render of a {self.cfg.guide_localization.object_name} with {self.cfg.guide_localization.prefix}yellow {self.cfg.guide_localization.edit}"
        logger.info(f"Localization prompt: {self.cfg.guide_localization.localization_prompt}")
        logger.info(f"Style prompt: {self.cfg.guide_localization.style_prompt}")
        logger.info(f"Background prompt: {self.cfg.guide_localization.background_prompt}")

    
    def calc_text_embeddings_csd(self, text) -> Union[torch.Tensor, List[torch.Tensor]]:
        text_z, text_z_neg = self.diffusion.encode_prompt(text)
        return text_z, text_z_neg

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]: # TODO - edit text embeddings
        if self.cfg.render.train_localization:
            text_z = {}
            if self.cfg.guide.prefix != "":
                self.cfg.guide.prefix += " "
            if self.cfg.guide.localization_prompt == "":
                text_z['localization_prompt'] = f"a 3d render of a {self.cfg.guide.text} wearing {self.cfg.guide.prefix}yellow {self.cfg.guide.edit}"
            if self.cfg.guide.style_prompt == "":
                if self.cfg.guide.style == '':
                    text_z['style_prompt'] = f"a 3d render of a {self.cfg.guide.text} wearing {self.cfg.guide.prefix}{self.cfg.guide.edit}"    
                else:
                    text_z['style_prompt'] = f"a 3d render of a {self.cfg.guide.text} wearing {self.cfg.guide.prefix}{self.cfg.guide.style} {self.cfg.guide.edit}"
            if self.cfg.guide.background_prompt == "":
                text_z['background_prompt'] = f"a 3d render of a {self.cfg.guide.text} wearing {self.cfg.guide.prefix}yellow {self.cfg.guide.edit}"
            logger.info(f"Localization prompt: {text_z['localization_prompt']}")
            logger.info(f"Style prompt: {text_z['style_prompt']}")
            logger.info(f"Background prompt: {text_z['background_prompt']}")
            if not self.cfg.guide.append_direction:
                text_z['localization_prompt'] = self.diffusion.get_text_embeds([text_z['localization_prompt']])
                text_z['style_prompt'] = self.diffusion.get_text_embeds([text_z['style_prompt']])
                text_z['background_prompt'] = self.diffusion.get_text_embeds([text_z['background_prompt']])
            else:
                text_z['localization_prompt_list'] = []
                text_z['style_prompt_list'] = []
                text_z['background_prompt_list'] = []
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    localization_prompt = f"{ text_z['localization_prompt']}, {d} view"
                    style_prompt = f"{text_z['style_prompt']}, {d} view"
                    background_prompt = f"{ text_z['background_prompt']}, {d} view"
                    text_z['localization_prompt_list'].append(self.diffusion.get_text_embeds([localization_prompt]))
                    text_z['style_prompt_list'].append(self.diffusion.get_text_embeds([style_prompt]))
                    text_z['background_prompt_list'].append(self.diffusion.get_text_embeds([background_prompt]))
                text_z['localization_prompt'] = text_z['localization_prompt_list']
                text_z['style_prompt'] = text_z['style_prompt_list']
                text_z['background_prompt'] = text_z['background_prompt_list']

        else:
            ref_text = self.cfg.guide.text
            if not self.cfg.guide.append_direction:
                text_z = self.diffusion.get_text_embeds([ref_text])
            else:
                text_z = []
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    text = f"{ref_text}, {d} view"
                    text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    def init_optimizer(self) -> Tuple[Optimizer, Any]:
        optimizer = torch.optim.Adam(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
        return optimizer, scaler
    
    def init_optimizer_csd(self):
        optim = torch.optim.Adam(self.nerf.get_params(self.cfg.optim.lr_csd), self.cfg.optim.lr_csd)
        return optim

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = NeRFDataset(self.cfg.render, device=self.device, type='train', H=self.cfg.render.train_h,
                                       W=self.cfg.render.train_w, size=100).dataloader()
        val_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                 W=self.cfg.render.eval_w,
                                 size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                       W=self.cfg.render.eval_w,
                                       size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_losses(self) -> Dict[str, Callable]:
        losses = {}
        if self.cfg.optim.lambda_shape > 0 and self.cfg.guide.shape_path is not None:
            from src.latent_nerf.training.losses.shape_loss import ShapeLoss
            losses['shape_loss'] = ShapeLoss(self.cfg.guide)
        if self.cfg.optim.lambda_sparsity > 0:
            from src.latent_nerf.training.losses.sparsity_loss import sparsity_loss
            losses['sparsity_loss'] = sparsity_loss
        return losses

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        if not self.cfg.render.train_localization: #TODO delete this line
            self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.nerf.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                if self.nerf.cuda_ray and self.train_step % self.cfg.render.update_extra_interval == 0:
                    with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                        self.nerf.update_extra_state()

                self.train_step += 1
                pbar.update(1)

                self.optimizer.zero_grad()

                if self.cfg.render.csd:
                    self.optim.zero_grad()
                    if self.cfg.guide_localization.anneal_t and (self.train_step > (self.cfg.optim.iters / 2)):
                        self.diffusion.update_step(min_step_percent=0.02, max_step_percent=0.5)
                        self.diffusion_II.update_step(min_step_percent=0.02, max_step_percent=0.5)


                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    if self.cfg.render.train_localization:
                        if not self.cfg.render.csd:
                            pred_rgbs, pred_rgbs_style, pred_rgbs_back, pred_prob, pred_rgb_style_origin, pred_rgb_back_origin, pred_ws, loss = self.train_render_localization(data)
                        else: 
                            pred_rgbs, pred_rgbs_style, pred_rgbs_back, pred_prob, pred_rgb_style_origin, pred_rgb_back_origin, pred_ws= self.train_render_csd(data)
                    else: 
                        pred_rgbs, pred_ws, loss = self.train_render(data)
                if not self.cfg.render.train_localization:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif not self.cfg.render.csd:
                    dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    self.scaler.scale(dummy_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    text_z = self.text_z.repeat(1, 1, 1) # [B, 77, 4096]
                    text_z_neg = self.text_z_neg.repeat(1, 1, 1) # [B, 77, 4096]
                    style_text_z = self.style_text_z.repeat(1, 1, 1) # [B, 77, 4096]
                    style_text_z_neg = self.style_text_z_neg.repeat(1, 1, 1) # [B, 77, 4096]
                    background_text_z = self.background_text_z.repeat(1, 1, 1) # [B, 77, 4096]
                    background_text_z_neg = self.background_text_z_neg.repeat(1, 1, 1) # [B, 77, 4096]
                    prob_sds = self.diffusion(pred_rgbs, text_z, text_z_neg)
                    style_sds = self.diffusion(pred_rgbs_style, style_text_z, style_text_z_neg)
                    background_sds = self.diffusion(pred_rgbs_back, background_text_z, background_text_z_neg)
                    # loss = style_sds['loss']
                    loss = prob_sds['loss'] + style_sds['loss'] + background_sds['loss']
                    
                    # if self.cfg.render.third_arc:
                    #     loss = prob_sds['loss'] + style_sds['loss']
                    if self.cfg.guide_localization.cascaded:
                        stage_I_loss = loss.clone()
                        prob_sds_II = self.diffusion_II(pred_rgbs, text_z, text_z_neg)
                        style_sds_II = self.diffusion_II(pred_rgbs_style, style_text_z, style_text_z_neg)
                        stage_II_loss = prob_sds_II['loss'] + style_sds_II['loss']
                        background_sds_II = self.diffusion_II(pred_rgbs_back, background_text_z, background_text_z_neg)
                        # if not self.cfg.render.third_arc:
                        #     stage_II_loss += background_sds_II['loss']
                        loss = stage_I_loss * self.cfg.guide_localization.stage_I_weight + stage_II_loss * self.cfg.guide_localization.stage_II_weight
                    loss.backward()
                    self.optim.step()



                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.nerf.train()

                if np.random.uniform(0, 1) < 0.05: #TODO
                # if True:
                    # Randomly log rendered images throughout the training
                    self.log_train_renders(pred_rgbs, 0)
                    if self.cfg.render.train_localization:
                        self.log_train_renders(pred_rgbs_style, 1)
                        self.log_train_renders(pred_rgbs_back, 2)
                        self.log_train_prob(pred_prob, 3)

        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        self.nerf.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
            all_preds_normals = []
            all_preds_depth = []
            if self.cfg.render.train_localization:
                all_preds_stylization = []
                all_preds_normals_stylization = []
                all_preds_background = []
                all_preds_normals_background = []

        for i, data in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                if self.cfg.render.train_localization:
                    preds_rgb, preds_rgb_style, preds_rgb_back, pred_rgb_mixed, preds_depth, preds_normals_localization, preds_normals_stylization, preds_normals_background = self.eval_render_localization(data)
                else:
                    preds, preds_depth, preds_normals = self.eval_render(data)
            if self.cfg.render.train_localization:
                pred_rgb = tensor2numpy(preds_rgb[0])
                pred_rgb_style = tensor2numpy(preds_rgb_style[0])
                pred_rgb_back = tensor2numpy(preds_rgb_back[0])
                pred_depth = tensor2numpy(preds_depth[0])
                pred_normals_localization = tensor2numpy(preds_normals_localization[0])
                pred_normals_stylization = tensor2numpy(preds_normals_stylization[0])
                pred_normals_background = tensor2numpy(preds_normals_background[0])
                pred_rgb_mixed = tensor2numpy(pred_rgb_mixed[0])

                if save_as_video:
                    all_preds.append(pred_rgb)
                    all_preds_stylization.append(pred_rgb_style)
                    all_preds_background.append(pred_rgb_back)
                    all_preds_normals.append(pred_normals_localization)
                    all_preds_normals_stylization.append(pred_normals_stylization)
                    all_preds_normals_background.append(pred_normals_background)
                    all_preds_depth.append(pred_depth)
                else:
                    if not self.cfg.log.skip_rgb:
                        Image.fromarray(pred_rgb).save(save_path / f"{self.train_step}_{i:04d}_rgb_localization.png")
                        Image.fromarray(pred_rgb_style).save(save_path / f"{self.train_step}_{i:04d}_rgb_stylization.png")
                        Image.fromarray(pred_rgb_back).save(save_path / f"{self.train_step}_{i:04d}_rgb_background.png")
                        Image.fromarray(pred_rgb_mixed).save(save_path / f"{self.train_step}_{i:04d}_pred_rgb_mixed.png")
                    Image.fromarray(pred_normals_localization).save(save_path / f"{self.train_step}_{i:04d}_normals_localization.png")
                    Image.fromarray(pred_normals_stylization).save(save_path / f"{self.train_step}_{i:04d}_normals_stylization.png")
                    Image.fromarray(pred_normals_background).save(save_path / f"{self.train_step}_{i:04d}_normals_background.png")
                    Image.fromarray(pred_depth).save(save_path / f"{self.train_step}_{i:04d}_depth.png")
            else:
                pred, pred_depth, pred_normals = tensor2numpy(preds[0]), tensor2numpy(preds_depth[0]), tensor2numpy(
                    preds_normals[0])

                if save_as_video:
                    all_preds.append(pred)
                    all_preds_normals.append(pred_normals)
                    all_preds_depth.append(pred_depth)
                else:
                    if not self.cfg.log.skip_rgb:
                        Image.fromarray(pred).save(save_path / f"{self.train_step}_{i:04d}_rgb.png")
                    Image.fromarray(pred_normals).save(save_path / f"{self.train_step}_{i:04d}_normals.png")
                    Image.fromarray(pred_depth).save(save_path / f"{self.train_step}_{i:04d}_depth.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_normals = np.stack(all_preds_normals, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            if self.cfg.render.train_localization:
                all_preds_stylization = np.stack(all_preds_stylization, axis=0)
                all_preds_normals_stylization = np.stack(all_preds_normals_stylization, axis=0)
                all_preds_background = np.stack(all_preds_background, axis=0)
                all_preds_normals_background = np.stack(all_preds_normals_background, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.train_step}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            if not self.cfg.log.skip_rgb:
                if self.cfg.render.train_localization:
                    dump_vid(all_preds, 'rgb_localization')
                    dump_vid(all_preds_stylization, 'rgb_stylization')
                    dump_vid(all_preds_background, 'rgb_background')
                else:
                    dump_vid(all_preds, 'rgb')
            if self.cfg.render.train_localization:
                dump_vid(all_preds_normals, 'normals_localization')
                dump_vid(all_preds_normals_stylization, 'normals_stylization')
                dump_vid(all_preds_normals_background, 'normals_background')
            else:
                dump_vid(all_preds_normals, 'normals')
            dump_vid(all_preds_depth, 'depth')
        logger.info('Done!')

    def full_eval(self):
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)

    def train_render_localization(self, data: Dict[str, Any]):
        rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        bg_color = torch.rand((B * N, 3), device=rays_o.device)  # Will be used if bg_radius <= 0
        outputs = self.nerf.render_localization(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True)
        pred_rgb = outputs['image_localization_mix'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_style = outputs['image_style_mix'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_back = outputs['image_back_mix'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        pred_prob = outputs['image_prob'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_style_origin = outputs['image_style_origin'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_back_origin = outputs['image_back_origin'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

        a = outputs['weights_sum'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # text embeddings
        # TODO - edit text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z_localization = self.text_z['localization_prompt'][dirs]
            text_z_stylization = self.text_z['style_prompt'][dirs]
            text_z_background = self.text_z['background_prompt'][dirs]
        else:
            text_z_localization = self.text_z['localization_prompt']
            text_z_stylization = self.text_z['style_prompt']
            text_z_background = self.text_z['background_prompt']


        # Guidance loss
        loss = self.diffusion.train_step(text_z_localization, pred_rgb)
        loss += self.diffusion.train_step(text_z_stylization, pred_rgb_style)
        loss += self.diffusion.train_step(text_z_background, pred_rgb_back)
        
        

        return pred_rgb, pred_rgb_style, pred_rgb_back, pred_prob, pred_rgb_style_origin, pred_rgb_back_origin, pred_ws, loss

    def train_render_csd(self, data: Dict[str, Any]):
        rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        bg_color = torch.rand((B * N, 3), device=rays_o.device)  # Will be used if bg_radius <= 0
        outputs = self.nerf.render_localization(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True)
        pred_rgb = outputs['image_localization_mix'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_style = outputs['image_style_mix'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # pred_rgb_style = outputs['image_style_origin'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_back = outputs['image_back_mix'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        pred_prob = outputs['image_prob'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_style_origin = outputs['image_style_origin'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_rgb_back_origin = outputs['image_back_origin'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

        a = outputs['weights_sum'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    
        return pred_rgb, pred_rgb_style, pred_rgb_back, pred_prob, pred_rgb_style_origin, pred_rgb_back_origin, pred_ws
    

    def train_render(self, data: Dict[str, Any]):
        rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        bg_color = torch.rand((B * N, 3), device=rays_o.device)  # Will be used if bg_radius <= 0
        outputs = self.nerf.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True)
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z

        # Guidance loss
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        loss = loss_guidance

        # Sparsity loss
        if 'sparsity_loss' in self.losses:
            loss += self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws)

        # Shape loss
        if 'shape_loss' in self.losses:
            loss += self.cfg.optim.lambda_shape * self.losses['shape_loss'](outputs['xyzs'], outputs['sigmas'])

        return pred_rgb, pred_ws, loss

    def eval_render_localization(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        bg_color = torch.ones(3, device=rays_o.device)  # [3]


        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.nerf.render_localization(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color)
        

        pred_depth = outputs['depth'].reshape(B, H, W)


        if self.nerf.latent_mode:
            pred_latent_localization = outputs['image_localization_mix'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            pred_latent_stylization = outputs['image_style_mix'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            pred_latent_background = outputs['image_back_mix'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent_localization.device)
                pred_rgb_style = torch.zeros(B, H, W, 3, device=pred_latent_stylization.device)
                pred_rgb_back = torch.zeros(B, H, W, 3, device=pred_latent_background.device)
            else:
                pred_rgb = self.diffusion.decode_latents(pred_latent_localization).permute(0, 2, 3, 1).contiguous()
                pred_rgb_style = self.diffusion.decode_latents(pred_latent_stylization).permute(0, 2, 3, 1).contiguous()
                pred_rgb_back = self.diffusion.decode_latents(pred_latent_background).permute(0, 2, 3, 1).contiguous()
        else:
            pred_rgb = outputs['image_localization_mix'].reshape(B, H, W, 3).contiguous().clamp(0, 1)
            pred_rgb_style = outputs['image_style_mix'].reshape(B, H, W, 3).contiguous().clamp(0, 1)
            pred_rgb_back = outputs['image_back_mix'].reshape(B, H, W, 3).contiguous().clamp(0, 1)
            pred_rgb_mixed = outputs['mixed_image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)

        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3)
        outputs_normals = self.nerf.render_localization(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                    disable_background=True)
        pred_normals_localization = outputs_normals['image_localization_mix'][:, :, :3].reshape(B, H, W, 3).contiguous()
        pred_normals_stylization = outputs_normals['image_style_mix'][:, :, :3].reshape(B, H, W, 3).contiguous()

        pred_normals_background = outputs_normals['image_back_mix'][:, :, :3].reshape(B, H, W, 3).contiguous()


        return pred_rgb, pred_rgb_style, pred_rgb_back, pred_rgb_mixed, pred_depth, pred_normals_localization, pred_normals_stylization, pred_normals_background

    def eval_render(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color)

        pred_depth = outputs['depth'].reshape(B, H, W)
        if self.nerf.latent_mode:
            pred_latent = outputs['image'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent.device)
            else:
                pred_rgb = self.diffusion.decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)

        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Render again for normals
        shading = 'normal'
        outputs_normals = self.nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                           ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                           disable_background=True)
        pred_normals = outputs_normals['image'][:, :, :3].reshape(B, H, W, 3).contiguous()

        return pred_rgb, pred_depth, pred_normals

    def log_train_prob(self, pred_rgbs: torch.Tensor, num):
        pred_image_vis = pred_rgbs.permute(0, 2, 3, 1).contiguous().clamp(0, 1)  # [1, H, W, 1]
        save_path = self.train_renders_path / f'step_{self.train_step:05d}_{num}.jpg'
        save_path.parent.mkdir(exist_ok=True)    
        pred = tensor2numpy(pred_image_vis[0])
        Image.fromarray(pred.squeeze(-1), mode='L').save(save_path)


    def log_train_renders(self, pred_rgbs: torch.Tensor, num):
        if self.nerf.latent_mode:
            pred_rgb_vis = self.diffusion.decode_latents(pred_rgbs).permute(0, 2, 3,
                                                                            1).contiguous()  # [1, 3, H, W]
        else:
            pred_rgb_vis = pred_rgbs.permute(0, 2, 3,
                                             1).contiguous().clamp(0, 1)  #
        save_path = self.train_renders_path / f'step_{self.train_step:05d}_{num}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred = tensor2numpy(pred_rgb_vis[0])

        Image.fromarray(pred).save(save_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        if not self.cfg.render.train_localization:
            checkpoint_dict = torch.load(checkpoint, map_location=self.device)
            if 'model' not in checkpoint_dict:
                self.nerf.load_state_dict(checkpoint_dict)
                logger.info("loaded model.")
                return

            missing_keys, unexpected_keys = self.nerf.load_state_dict(checkpoint_dict['model'], strict=False)
            logger.info("loaded model.")
            if len(missing_keys) > 0:
                logger.warning(f"missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.warning(f"unexpected keys: {unexpected_keys}")
        else:
            checkpoint_dict = torch.load(self.cfg.render.nerf_path, map_location=self.device)

            self.nerf.sigma_net.load_state_dict(checkpoint_dict['sigma_net'], strict=False)
            self.nerf.encoder.load_state_dict(checkpoint_dict['encoder'], strict=False)
            if self.nerf.decoder_layer is not None:
                if checkpoint_dict['decoder'] is not None:
                    self.nerf.decoder_layer.load_state_dict(checkpoint_dict['decoder'], strict=False)
                else:
                    self.nerf.decoder_layer = None
                    
            logger.info("loaded localization model.")
        if self.cfg.render.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.nerf.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.nerf.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict and not self.cfg.render.train_localization:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

        if self.scaler and 'scaler' in checkpoint_dict and not self.cfg.render.train_localization:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("loaded scaler.")
            except:
                logger.warning("Failed to load scaler.")

    def save_checkpoint(self, full=False):

        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if self.nerf.cuda_ray:
            state['mean_count'] = self.nerf.mean_count
            state['mean_density'] = self.nerf.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()

        state['model'] = self.nerf.state_dict()
        # TODO
        state['sigma_net'] = self.nerf.sigma_net.state_dict()
        state['encoder'] = self.nerf.encoder.state_dict()
        if self.nerf.decoder_layer is not None:
            state['decoder'] = self.nerf.decoder_layer.state_dict()

        file_path = f"{name}.pth"

        self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)
