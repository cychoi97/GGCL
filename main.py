import os
import torch
import random
import argparse
import numpy as np

from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # Set random seed for reproducibility
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("Random Seed: ", seed)

    # Create directories if not exist.
    log_dir = os.path.join(config.save_path, 'logs')
    model_save_dir = os.path.join(config.save_path, 'models')
    sample_dir = os.path.join(config.save_path, 'samples')
    png_result_dir = os.path.join(config.save_path, 'results/png')
    dcn_result_dir = os.path.join(config.save_path, 'results/dcm')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(png_result_dir, exist_ok=True)
    os.makedirs(dcn_result_dir, exist_ok=True)

    # Solver for training and testing.
    if config.mode == 'train':
        # Train data loader.
        siemens_loader = get_loader(config.batch_size, config.root_path, dataset='SIEMENS',
                                    image_size=config.image_size, mode=config.mode,
                                    shuffle=True, num_workers=config.num_workers)
        ge_loader = get_loader(config.batch_size, config.root_path, dataset='GE',
                               image_size=config.image_size, mode=config.mode,
                               shuffle=True, num_workers=config.num_workers)

        # Valid data loader.
        valid_siemens_loader = get_loader(1, config.root_path, dataset='SIEMENS',
                                          image_size=config.image_size, mode='valid',
                                          shuffle=False, num_workers=0)
        valid_ge_loader = get_loader(1, config.root_path, dataset='GE',
                                     image_size=config.image_size, mode='valid',
                                     shuffle=False, num_workers=0)

        solver = Solver(siemens_loader, ge_loader, valid_siemens_loader, valid_ge_loader, config)
        if config.dataset in ['SIEMENS', 'GE']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        # Test data loader.
        test_siemens_loader = get_loader(1, config.root_path, dataset='SIEMENS',
                                         image_size=config.image_size, mode=config.mode,
                                         shuffle=False, num_workers=0)
        test_ge_loader = get_loader(1, config.root_path, dataset='GE',
                                    image_size=config.image_size, mode=config.mode,
                                    shuffle=False, num_workers=0)

        solver = Solver(None, None, test_siemens_loader, test_ge_loader, config)
        if config.dataset in ['SIEMENS', 'GE']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # multi gpus
    parser.add_argument('--multi-gpu-mode', type=str, default='Single',
                        choices=['Single', 'DataParallel'], help='multi-gpu-mode')
  
    # Model configuration.
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--c1_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=1, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--image_size', type=int, default=512, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=7, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_ggcl', type=float, default=2, help='weight for ggcl loss')
    parser.add_argument('--use_feature', action='store_true', help='If specified, use GGDR or GGCL')
    parser.add_argument('--guide_type', type=str, default='ggcl',
                        choices=['ggdr', 'ggcl'], help='choose between GGDR and GGCL')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='SIEMENS', choices=['SIEMENS', 'GE', 'Both'])
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=400000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=200000, help='number of iterations for decaying lr')
    parser.add_argument('--num_patches', type=int, default=64, help='number of patch for GGCL')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models.py for more details.')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=400000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--dicom_save', action='store_true', help='If specified, dicom result will be saved.')

    # Directories.
    parser.add_argument('--root_path', type=str, help="your training dataset path", required=True)
    parser.add_argument('--save_path', type=str, default='./result')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)