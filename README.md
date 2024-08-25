# paintbrushPP

for deleting the lock that bothers the build:
rm -rf ~/.cache/torch_extensions/


for resources:
srun -p threedle-contrib --pty --gres=gpu:1 --mem-per-cpu=128G /bin/bash

for gpu memory
nvidia-smi
