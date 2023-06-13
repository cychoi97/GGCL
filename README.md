# CT Kernel Conversion Using Multi-Domain Image Translation with Generator-Guided Contrastive Learning

* Code description will be updated soon..


## Dependencies

* CUDA 11.6
* Pytorch 1.10.0

Please install [Pytorch](https://pytorch.org/) for your own CUDA version.

Also, install the other packages in `requirements.txt` following:
```bash
pip install -r requirements.txt
```

### Prepare your own dataset

For example, you should set dataset path following:
```text
root_path
    ├── train
    |     ├── SIEMENS
    |     |     ├── B30f
    |     |     |     ├── 0001.dcm
    |     |     |     ├── 0002.dcm
    |     |     |     └── 0003.dcm
    |     |     ├── B50f
    |     |     └── B70f
    |     └── GE
    |          ├── SOFT
    |          ├── CHEST
    |          └── EDGE
    ├── valid
    └── test
```


## Training

For multi-GPU, you can use `--multi-gpu-mode DataParallel`.

### Train

For one dataset (e.g. SIEMENS),
```python
python main.py --mode train --dataset SIEMENS --batch_size 2 --root_path 'your_own_dataset_path' --use_feature --guide_type ggcl
```
for two dataset (e.g. SIEMENS and GE),
```python
python main.py --mode train --dataset Both --batch_size 2 --root_path 'your_own_dataset_path' --use_feature --guide_type ggcl
```

For generator-guided contrastive learning, you have to specify two arguments below:

* `--use_feature` makes GGCL (or [GGDR](https://github.com/naver-ai/GGDR)) run.
* `--guide_type` decides which regularization method is used between GGDR and GGCL. Default is GGCL.

Model checkpoints and validation samples will be stored in ./result/models and ./result/samples, respectively.

### resume

To restart training, you can use `--resume_iters`.
```python
python main.py --mode train --dataset SIEMENS --batch_size 2 --root_path 'your_own_dataset_path' --use_feature --guide_type ggcl --resume_iters 100000
```


## Test




## Acknowledgement

Our code is based on [StarGAN](https://github.com/yunjey/stargan), and `data_loader.py` is inspired by [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
