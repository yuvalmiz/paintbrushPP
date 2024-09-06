# Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures - localizaed painting
<a href="https://arxiv.org/abs/2211.07600"><img src="https://img.shields.io/badge/arXiv-2211.07600-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  

In this repository we utilize the structure of latent-nerf code in order to do 2 things. first we will use latent-paint which resides inside this repository to evaluate our 3d-paintbrush csd function and second we will use it to create NeRF and make an archtecture to create localized edits in NeRF.


## Description :scroll:	
Official Implementation for "Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures".

> TL;DR - We explore different ways of introducing shape-guidance for Text-to-3D and present three models: a purely text-guided Latent-NeRF, Latent-NeRF with soft shape guidance for more exact control over the generated shape, and Latent-Paint for texture generation for explicit shapes.

by using the Latent-NeRF with soft shape guidence we create NeRF's to work with in order to create localized editing


### Latent-Paint

In the `Latent-Paint` application, a texture is generated for an explicit mesh directly on its texture map using stable-diffusion as a prior.

Here the geometry is used as a hard constraint where the generation process is tied to the given mesh and its parameterization.

<img src="https://github.com/eladrich/latent-nerf/raw/docs/docs/fish.gif" width="800px"/>

To create such results, run the `train_latent_paint` script. Parameters are handled using [pyrallis](https://eladrich.github.io/pyrallis/) and can be passed from a config file or the cmd.

```bash
 python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/goldfish.yaml
```

or alternatively 

```bash
python -m scripts.train_latent_paint --log.exp_name 2022_11_22_goldfish --guide.text "A goldfish"  --guide.shape_path /nfs/private/gal/meshes/blub.obj
```

### Latent-NeRF using a soft guidance shape:

Here we use a simple coarse geometry which we call a `SketchShape` to guide the generation process. 

A `SketchShape` presents a soft constraint which guides the occupancy of a learned NeRF model but isn't constrained to its exact geometry.

in order to make localized editing we will first create a NeRF to work on.
to create a NeRF that works in latent space:
```
python -m scripts.train_latent_nerf –log.exp_name "nascar" –guide.text "a next gen nascar" –guide.shape_path "shapes/nascar.obj" –render.nerf_type latent
```

To create NeRF that works in rgb:

```
python -m scripts.train_latent_nerf –log.exp_name "nascar" –guide.text "a next gen nascar" –guide.shape_path "shapes/nascar.obj" –render.nerf_type rgb 
```


### Unconstrained Latent-NeRF :
You can also create a NeRF without using any shape guide. for the localized editing of the NeRF it doesnt metter.

<p align="left">
  <img src="https://github.com/eladrich/latent-nerf/raw/docs/docs/castle.gif" width="200px"/>
  <img src="https://github.com/eladrich/latent-nerf/raw/docs/docs/palmtree.gif" width="200px"/>
  <img src="https://github.com/eladrich/latent-nerf/raw/docs/docs/fruits.gif" width="200px"/>
  <img src="https://github.com/eladrich/latent-nerf/raw/docs/docs/pancake.gif" width="200px"/>
</p>

To create such results, run the `train_latent_nerf` script. Parameters are handled using [pyrallis](https://eladrich.github.io/pyrallis/) and can be passed from a config file or the cmd.

```bash
 python -m scripts.train_latent_nerf --config_path demo_configs/latent_nerf/sand_castle.yaml
```

Or alternatively
```bash
python -m scripts.train_latent_nerf --log.exp_name 'sand_castle' --guide.text 'a highly detailed sand castle' --render.nerf_type latent
```

```bash
python -m scripts.train_latent_nerf --log.exp_name 'sand_castle' --guide.text 'a highly detailed sand castle' --render.nerf_type rgb
```

## localized editing:
In order to make a localized editing first make sure you create a NeRF in the right space (RGB/Latent).

Our code is working on RGB thus we will work in this space. it is possible for future works to use latent space.

To run:
```
python -m scripts.train_latent_nerf --log.exp_name "<output directory name>" --guide.text "<text that describes the NeRF>" --guide_localization.object_name "<object name>" --guide_localization.style "<style of the localized object you want to edit>" --guide_localization.edit "<the localized object>" --render.nerf_type "rgb" --render.train_localization "True" --render.nerf_path "<path to your NeRF>" --render.cuda_ray "False" --render.csd "True"
```
there are 3 architectures
without any parameter it will work on the first architecture
with --render.first_arc "True" it will work with the second architecture described in the project report
for the third archtecture use --render.second_arc "True"
in order to work with the diffusion model that we work with in 3d-paintbrush use --render.csd "True", of you dont use it it will work with stable-diffusion

example of usage:
1. create NeRF:
python -m scripts.train_latent_nerf –log.exp_name "bunny" –guide.text "a bunny" –guide.shape_path "shapes/bunny.obj" –render.nerf_type rgb 

2. use edit: assume the location of the NeRF is at: /net/projects/ranalab/yuvalm_amitd/paintbrushPP/experiments/bunny/checkpoints/step_005000.pth

```
python -m scripts.train_latent_nerf --log.exp_name "bunny_second_arc" --guide.text "a bunny" --guide_localization.object_name "bunny" --guide_localization.style "golden" --guide_localization.edit "necklace" --render.nerf_type "rgb" --render.train_localization "True" --render.nerf_path "/net/projects/ranalab/yuvalm_amitd/paintbrushPP/experiments/bunny/checkpoints/step_005000.pth" --render.first_arc "True" --render.cuda_ray “False” --render.csd "True"
```

## Getting Started


### Installation :floppy_disk:	
Install the common dependencies from the `requirements.txt` file
```bash
pip install -r requirements.txt
```

For `Latent-NeRF` with shape-guidance, additionally install `igl`
```bash
conda install -c conda-forge igl
```

For `Latent-Paint`, additionally install `kaolin`
```bash
 pip install git+https://github.com/NVIDIAGameWorks/kaolin
```


Note that you also need a :hugs: token for StableDiffusion and DeepFloyd models. First accept conditions for the models ( https://huggingface.co/CompVis/stable-diffusion-v1-4, https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). Then, add a TOKEN file [access token](https://huggingface.co/settings/tokens) to the root folder of this project, or use the `huggingface-cli login` command


### Additional Tips and Tricks :magic_wand:	

* Check out the `vis/train` to see the actual rendering used during the optimization. You might want to play around with the `guide.mesh_scale` if the object looks too small or too large. 

* For `Latent-NeRF` with shape-guidance try changing `guide.proximal_surface` and `optim.lambda_shape` to control the strictness of the guidance


## important commands
sometimes when playing with the cpp files when building then there is a lock the bothers the build
for deleting the that lock:
```
rm -rf ~/.cache/torch_extensions/
```

for gpu memory
```
nvidia-smi
```


## Acknowledgments
The `Latent-NeRF` code is heavily based on the [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) project, and the `Latent-Paint` code borrows from [text2mesh](https://github.com/threedle/text2mesh).

## Citation
If you use this code for your research, please cite our paper [Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures](https://arxiv.org/abs/2211.07600)

```
@article{metzer2022latent,
  title={Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures},
  author={Metzer, Gal and Richardson, Elad and Patashnik, Or and Giryes, Raja and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2211.07600},
  year={2022}
}
```


