# CT Kernel Conversion Using Multi-Domain Image Translation with Generator-Guided Contrastive Learning

* Code description will be updated soon..


## Requirements

```bash
pip install -r requirements.txt
```

### Prepare your own dataset

For example, You should set dataset path, as follows:
 ```text
  root_path
      ├── train
            ├── SIEMENS
                   ├── B30f
                   ├── B50f
                   └── B70f
            └── GE
                 ├── SOFT
                 ├── CHEST
                 └── EDGE
      ├── valid
      └── test
  ```

## Acknowledgement

Our code is based on [StarGAN](https://github.com/yunjey/stargan), and `data_loader.py` is inspired by [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
