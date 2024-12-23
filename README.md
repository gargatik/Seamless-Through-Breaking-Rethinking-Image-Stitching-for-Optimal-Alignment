# Stitching
download our paper https://github.com/gargatik/Seamless-Through-Breaking-Rethinking-Image-Stitching-for-Optimal-Alignment/blob/master/Stitching_Through_Breaking.pdf

## Dataset (UDIS-D)
We use the UDIS-D dataset to train and evaluate our method. Please refer to [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for more details about this dataset.

## Requirements
```Shell
conda create -y --name stitching
conda activate stitching
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install matplotlib tensorboard scipy 
pip install diffusers["torch"]==0.21.4 transformers
pip install yacs loguru einops timm==0.4.12 imageio wandb scikit-image==0.19.3
# install mmcv by  https://mmcv.readthedocs.io/en/latest/get_started/installation.html
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install opencv-contrib-python

# if in docker container : 
pip uninstall opencv-python
pip install opencv-python-headless
pip uninstall opencv-contrib-python
pip install opencv-contrib-python-headless
```

## Evaluation
```Shell
python evaluate.py --data_dir ./data/UDIS/UDIS-D/
```

## INFRENCE
```Shell
python out.py  --data_root_path ./demo/ --inf_cfg "all_img1_with_inpaint_g12_transRef"

# You can add custom inference config at inf_configs folder.   e.g. ADD inf_configs/mycustom.py
python out.py  --data_root_path ./demo/ --inf_cfg  "mycustom"
```

## Acknowledgement
In this project, we use parts of codes in:
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
- [UDIS++](https://github.com/nie-lang/udis2)
- [TransRef](https://github.com/Cameltr/TransRef)
- [diffuser](https://github.com/huggingface/diffusers)
