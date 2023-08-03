import os
import time
import pydicom
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from packaging import version

from model import (
    Generator,
    Discriminator,
    Generator_GGCL,
    Discriminator_GGCL
    )

from torchvision.utils import save_image


class PatchNCELoss(nn.Module):
    def __init__(self, batch_size, nce_includes_all_negatives_from_minibatch):
        super().__init__()
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07 # temperature nce

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, siemens_loader, ge_loader, valid_siemens_loader, valid_ge_loader, config):
        """Initialize configurations."""
        # Multi gpu
        self.multi_gpu_mode = config.multi_gpu_mode

        # Data loader.
        self.siemens_loader  = siemens_loader
        self.ge_loader = ge_loader

        self.valid_siemens_loader = valid_siemens_loader
        self.valid_ge_loader = valid_ge_loader

        # Model configurations.
        self.c1_dim = config.c1_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_ggcl = config.lambda_ggcl
        self.use_feature = config.use_feature
        self.guide_type = config.guide_type

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_patches = config.num_patches
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.mode = config.mode
        self.nce_includes_all_negatives_from_minibatch = config.nce_includes_all_negatives_from_minibatch

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.dicom_save = config.dicom_save
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.root_path = config.root_path
        self.save_path = config.save_path
        self.log_dir = os.path.join(self.save_path, 'logs')
        self.sample_dir = os.path.join(self.save_path, 'samples')
        self.model_save_dir = os.path.join(self.save_path, 'models')
        self.result_dir = os.path.join(self.save_path, 'results')

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if not self.use_feature:
            if self.dataset in ['SIEMENS']:
                self.G = Generator(self.g_conv_dim, self.c1_dim, self.g_repeat_num)
                self.D = Discriminator(self.image_size, self.d_conv_dim, self.c1_dim, self.d_repeat_num)
            elif self.dataset in ['GE']:
                self.G = Generator(self.g_conv_dim, self.c2_dim, self.g_repeat_num)
                self.D = Discriminator(self.image_size, self.d_conv_dim, self.c2_dim, self.d_repeat_num)
            elif self.dataset in ['Both']:
                self.G = Generator(self.g_conv_dim, self.c1_dim+self.c2_dim+2, self.g_repeat_num)
                self.D = Discriminator(self.image_size, self.d_conv_dim, self.c1_dim+self.c2_dim, self.d_repeat_num)
        else:
            if self.dataset in ['SIEMENS']:
                self.G = Generator_GGCL(self.g_conv_dim, self.c1_dim, self.g_repeat_num)
                self.D = Discriminator_GGCL(self.image_size, self.d_conv_dim, self.c1_dim, self.d_repeat_num)
            elif self.dataset in ['GE']:
                self.G = Generator_GGCL(self.g_conv_dim, self.c2_dim, self.g_repeat_num)
                self.D = Discriminator_GGCL(self.image_size, self.d_conv_dim, self.c2_dim, self.d_repeat_num)
            elif self.dataset in ['Both']:
                self.G = Generator_GGCL(self.g_conv_dim, self.c1_dim+self.c2_dim+2, self.g_repeat_num)
                self.D = Discriminator_GGCL(self.image_size, self.d_conv_dim, self.c1_dim+self.c2_dim, self.d_repeat_num)

        if self.multi_gpu_mode == 'DataParallel':
            print("Multi GPU model = DataParallel")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print(f'The number of parameters: {num_params}')

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print(f'Loading the trained models from step {resume_iters}...')
        G_path = os.path.join(self.model_save_dir, f'{resume_iters}-G.ckpt')
        D_path = os.path.join(self.model_save_dir, f'{resume_iters}-D.ckpt')

        #### Multi-GPU
        if self.multi_gpu_mode == 'DataParallel':
            print("Multi GPU model = DataParallel")
            self.G.module.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.module.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        else:
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.contiguous().view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=3):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def cosine_distance_loss(self, x, y):
        return 1. - F.cosine_similarity(x, y).mean()
    
    def L2_norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm + 1e-7)
        return out

    def patchsample(self, feat, num_patches=256, patch_id=None):
        B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if num_patches > 0:
            if patch_id is not None:
                    patch_id = patch_id
            else:
                patch_id = np.random.permutation(feat_reshape.shape[1])
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
            patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        else:
            x_sample = feat_reshape
            patch_id = []
        x_sample = self.L2_norm(x_sample)

        if num_patches == 0:
            x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
        return x_sample, patch_id

    def PatchNCE_loss(self, src, tgt):
        self.criterionNCE = PatchNCELoss(self.batch_size, self.nce_includes_all_negatives_from_minibatch)
        feat_k_pool, sample_ids = self.patchsample(src, self.num_patches, None) # output size: (512, 128)
        feat_q_pool, _ = self.patchsample(tgt, self.num_patches, sample_ids)
        loss = self.criterionNCE(feat_q_pool, feat_k_pool) # weight NCE loss
        loss = loss.mean()
        return loss
    
    def save_dicom(self, dcm_path, predict_output, save_path):
        predict_img = predict_output.copy()
        dcm = pydicom.dcmread(dcm_path, force=True)

        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        
        predict_img -= np.float32(intercept)
        if slope != 1:
            predict_img = predict_img.astype(np.float32) / slope
        predict_img = predict_img.astype(np.int16)

        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dcm.PixelData = predict_img.squeeze().tobytes()
        dcm.SmallestImagePixelValue = predict_img.min()
        dcm.LargestImagePixelValue = predict_img.max()

        dcm[0x0028,0x0106].VR = 'US'
        dcm[0x0028,0x0107].VR = 'US'

        dcm.save_as(save_path)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.siemens_loader if self.dataset == 'SIEMENS' else self.ge_loader
        data_iter = iter(data_loader)

        # Fetch fixed inputs for debugging.
        valid_iter = iter(self.valid_siemens_loader) \
            if self.dataset == 'SIEMENS' else iter(self.valid_ge_loader)
        data_dict = next(valid_iter)
        x_fixed = data_dict['image']
        c_org = data_dict['label']
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c1_dim) \
            if self.dataset == 'SIEMENS' else self.create_labels(c_org, self.c2_dim)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                data_dict = next(data_iter)
                x_real = data_dict['image']
                label_org = data_dict['label']
            except:
                data_iter = iter(data_loader)
                data_dict = next(data_iter)
                x_real = data_dict['image']
                label_org = data_dict['label']

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'SIEMENS':
                c_org = self.label2onehot(label_org, self.c1_dim)
                c_trg = self.label2onehot(label_trg, self.c1_dim)
            elif self.dataset == 'GE':
                c_org = self.label2onehot(label_org, self.c2_dim)
                c_trg = self.label2onehot(label_trg, self.c2_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            if not self.use_feature:
                out_src, out_cls = self.D(x_real)
            else:
                out_src, out_cls, _ = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            if not self.use_feature:
                x_fake = self.G(x_real, c_trg)
                out_src, _  = self.D(x_fake.detach())
            else:
                x_fake, g_out_feature = self.G(x_real, c_trg)
                out_src, _, d_out_feature  = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            if not self.use_feature:
                out_src, _ = self.D(x_hat)
            else:
                out_src, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Compute loss for GGCL.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            if self.use_feature:
                if self.guide_type == 'ggdr':
                    d_loss_ggcl = self.cosine_distance_loss(g_out_feature, d_out_feature)
                elif self.guide_type == 'ggcl':
                    d_loss_ggcl = self.PatchNCE_loss(g_out_feature, d_out_feature)
                d_loss = d_loss + self.lambda_ggcl*d_loss_ggcl

            # Backward and optimize.
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            if self.use_feature:
                loss['D/loss_ggcl'] = d_loss_ggcl.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                if not self.use_feature:
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                else:
                    x_fake, _ = self.G(x_real, c_trg)
                    out_src, out_cls, _ = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                if not self.use_feature:
                    x_reconst = self.G(x_fake, c_org)
                else:
                    x_reconst, _ = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training info.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = f'Elapsed [{et}], Iteration [{i+1}/{self.num_iters}], Dataset [{self.dataset}]'
                for tag, value in loss.items():
                    log += f', {tag} : {value:.5f}'
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        if not self.use_feature:
                            fake = self.G(x_fixed, c_fixed)
                        else:
                            fake, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(fake)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, f'{i+1}-images.png')
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print(f'Saved real and fake images into {sample_path}...')

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, f'{i+1}-G.ckpt')
                D_path = os.path.join(self.model_save_dir, f'{i+1}-D.ckpt')
                if hasattr(self.G, 'module'):
                    torch.save(self.G.module.state_dict(), G_path)
                    torch.save(self.D.module.state_dict(), D_path)
                else:
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                print(f'Saved model checkpoints into {self.model_save_dir}...')

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print (f'Decayed learning rates, g_lr: {g_lr}, d_lr: {d_lr}.')

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        siemens_iter = iter(self.siemens_loader)
        ge_iter = iter(self.ge_loader)

        # Fetch fixed inputs for debugging.
        valid_siemens_iter = iter(self.valid_siemens_loader)
        data_dict = next(valid_siemens_iter)
        x_fixed = data_dict['image']
        x_fixed = x_fixed.to(self.device)
        c_org = data_dict['label']
        c_siemens_list = self.create_labels(c_org, self.c1_dim)
        c_ge_list = self.create_labels(c_org, self.c2_dim)
        zero_siemens = torch.zeros(x_fixed.size(0), self.c1_dim).to(self.device)
        zero_ge = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)
        mask_siemens = self.label2onehot(torch.zeros(x_fixed.size(0)), dim=2).to(self.device)
        mask_ge = self.label2onehot(torch.ones(x_fixed.size(0)), dim=2).to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['SIEMENS', 'GE']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = siemens_iter if dataset == 'SIEMENS' else ge_iter

                try:
                    data_dict = next(data_iter)
                    x_real = data_dict['image']
                    label_org = data_dict['label']
                except:
                    if dataset == 'SIEMENS':
                        siemens_iter = iter(self.siemens_loader)
                        data_dict = next(siemens_iter)
                        x_real = data_dict['image']
                        label_org = data_dict['label']
                    elif dataset == 'GE':
                        ge_iter = iter(self.ge_loader)
                        data_dict = next(ge_iter)
                        x_real = data_dict['image']
                        label_org = data_dict['label']

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'SIEMENS':
                    c_org = self.label2onehot(label_org, self.c1_dim)
                    c_trg = self.label2onehot(label_trg, self.c1_dim)
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), dim=2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'GE':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c1_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), dim=2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)         # Input images.
                c_org = c_org.to(self.device)           # Original domain labels.
                c_trg = c_trg.to(self.device)           # Target domain labels.
                label_org = label_org.to(self.device)   # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)   # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                if not self.use_feature:
                    out_src, out_cls = self.D(x_real)
                else:
                    out_src, out_cls, _ = self.D(x_real)
                out_cls = out_cls[:, :self.c1_dim] if dataset == 'SIEMENS' else out_cls[:, self.c1_dim:]
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org)

                # Compute loss with fake images.
                if not self.use_feature:
                    x_fake = self.G(x_real, c_trg)
                    out_src, _  = self.D(x_fake.detach())
                else:
                    x_fake, g_out_feature = self.G(x_real, c_trg)
                    out_src, _, d_out_feature = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha*x_real.data + (1-alpha)*x_fake.data).requires_grad_(True)
                if not self.use_feature:
                    out_src, _ = self.D(x_hat)
                else:
                    out_src, _, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Compute loss for GGCL.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls*d_loss_cls + self.lambda_gp*d_loss_gp
                if self.use_feature:
                    if self.guide_type == 'ggdr':
                        d_loss_ggcl = self.cosine_distance_loss(g_out_feature, d_out_feature)
                    elif self.guide_type == 'ggcl':
                        d_loss_ggcl = self.PatchNCE_loss(g_out_feature, d_out_feature)
                    d_loss = d_loss + self.lambda_ggcl*d_loss_ggcl
                
                # Backward and optimize.
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                if self.use_feature:
                    loss['D/loss_ggcl'] = d_loss_ggcl.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    if not self.use_feature:
                        x_fake = self.G(x_real, c_trg)
                        out_src, out_cls = self.D(x_fake)
                    else:
                        x_fake, _ = self.G(x_real, c_trg)
                        out_src, out_cls, _ = self.D(x_fake)
                    out_cls = out_cls[:, :self.c1_dim] if dataset == 'SIEMENS' else out_cls[:, self.c1_dim:]
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg)

                    # Target-to-original domain.
                    if not self.use_feature:
                        x_reconst = self.G(x_fake, c_org)
                    else:
                        x_reconst, _ = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_cls*g_loss_cls + self.lambda_rec*g_loss_rec
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                    loss['G/loss_rec'] = g_loss_rec.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = f'Elapsed [{et}], Iteration [{i+1}/{self.num_iters}], Dataset [{dataset}]'
                    for tag, value in loss.items():
                        log += f', {tag} : {value:.5f}'
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_siemens_list:
                        c_trg = torch.cat([c_fixed, zero_ge, mask_siemens], dim=1)
                        if not self.use_feature:
                            fake = self.G(x_fixed, c_trg)
                        else:
                            fake, _ = self.G(x_fixed, c_trg)
                        x_fake_list.append(fake)
                    for c_fixed in c_ge_list:
                        c_trg = torch.cat([zero_siemens, c_fixed, mask_ge], dim=1)
                        if not self.use_feature:
                            fake = self.G(x_fixed, c_trg)
                        else:
                            fake, _ = self.G(x_fixed, c_trg)
                        x_fake_list.append(fake)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, f'{i+1}-images.png')
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print(f'Saved real and fake images into {sample_path}...')

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, f'{i+1}-G.ckpt')
                D_path = os.path.join(self.model_save_dir, f'{i+1}-D.ckpt')
                if hasattr(self.G, 'module'):
                    torch.save(self.G.module.state_dict(), G_path)
                    torch.save(self.D.module.state_dict(), D_path)
                else:
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                print(f'Saved model checkpoints into {self.model_save_dir}...')

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print(f'Decayed learning rates, g_lr: {g_lr}, d_lr: {d_lr}.')

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'SIEMENS':
            data_loader = self.valid_siemens_loader
        elif self.dataset == 'GE':
            data_loader = self.valid_ge_loader
        
        with torch.no_grad():
            for i, data_dict in enumerate(data_loader):
                x_real = data_dict['image']
                c_org = data_dict['label']
                path = data_dict['path'][0]

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                if self.dataset == 'SIEMENS':
                    c_trg_list = self.create_labels(c_org, self.c1_dim)
                elif self.dataset == 'GE':
                    c_trg_list = self.create_labels(c_org, self.c2_dim)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    if not self.use_feature:
                        fake = self.G(x_real, c_trg)
                    else:
                        fake, _ = self.G(x_real, c_trg)
                    x_fake_list.append(fake)

                    # save as dicom
                    if self.dicom_save:
                        predict = (self.denorm(fake.data.cpu())*4095.0-1024.0).numpy().astype(np.float32)
                        if self.dataset == 'SIEMENS':
                            dcm_save_path = os.path.join(self.result_dir, f'dcm/{i+1}_SIEMENS_{str(c_org.numpy())}_to_SIEMENS_{c_trg.cpu()}.dcm')
                        elif self.dataset == 'GE':
                            dcm_save_path = os.path.join(self.result_dir, f'dcm/{i+1}_GE_{str(c_org.numpy())}_to_GE_{c_trg.cpu()}.dcm')
                        self.save_dicom(path, predict, dcm_save_path)
                        print(f'Saved fake dicom image into {dcm_save_path}...')

                if self.dataset == 'SIEMENS':
                    png_result_path = os.path.join(self.result_dir, f'png/{i+1}_SIEMENS_{str(c_org.numpy())}.png')
                elif self.dataset == 'GE':
                    png_result_path = os.path.join(self.result_dir, f'png/{i+1}_GE_{str(c_org.numpy())}.png')

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                save_image(self.denorm(x_concat.data.cpu()), png_result_path, nrow=1, padding=0)
                print(f'Saved real and fake images into {png_result_path}...')

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            for num, loader in enumerate([self.valid_siemens_loader, self.valid_ge_loader]):
                for i, data_dict in enumerate(loader):
                    x_real = data_dict['image']
                    c_org = data_dict['label']
                    path = data_dict['path'][0]

                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)
                    c_siemens_list = self.create_labels(c_org, self.c1_dim)
                    c_ge_list = self.create_labels(c_org, self.c2_dim)
                    zero_siemens = torch.zeros(x_real.size(0), self.c1_dim).to(self.device)                 # Zero vector for SIEMENS.
                    zero_ge = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)                      # Zero vector for GE.
                    mask_siemens = self.label2onehot(torch.zeros(x_real.size(0)), dim=2).to(self.device)    # Mask vector: [1, 0].
                    mask_ge = self.label2onehot(torch.ones(x_real.size(0)), dim=2).to(self.device)          # Mask vector: [0, 1].

                    x_fake_list = [x_real]
                    for c_siemens in c_siemens_list:
                        c_trg = torch.cat([c_siemens, zero_ge, mask_siemens], dim=1)
                        if not self.use_feature:
                            fake = self.G(x_real, c_trg)
                        else:
                            fake, _ = self.G(x_real, c_trg)
                        x_fake_list.append(fake)

                        # Save as dicom
                        if self.dicom_save:
                            predict = (self.denorm(fake.data.cpu())*4095.0-1024.0).numpy().astype(np.float32)
                            if num == 0:
                                dcm_save_path = os.path.join(self.result_dir, f'dcm/{i+1}_SIEMENS_{str(c_org.numpy())}_to_SIEMENS_{c_siemens.cpu()}.dcm')
                                self.save_dicom(path, predict, dcm_save_path)
                            elif num == 1:
                                dcm_save_path = os.path.join(self.result_dir, f'dcm/{i+1}_GE_{str(c_org.numpy())}_to_SIEMENS_{c_siemens.cpu()}.dcm')
                                self.save_dicom(path, predict, dcm_save_path)
                            print(f'Saved fake dicom image into {dcm_save_path}...')
                    for c_ge in c_ge_list:
                        c_trg = torch.cat([zero_siemens, c_ge, mask_ge], dim=1)
                        if not self.use_feature:
                            fake = self.G(x_real, c_trg)
                        else:
                            fake, _ = self.G(x_real, c_trg)
                        x_fake_list.append(fake)

                        # Save as dicom
                        if self.dicom_save:
                            predict = (self.denorm(fake.data.cpu())*4095.0-1024.0).numpy().astype(np.float32)
                            if num == 0:
                                dcm_save_path = os.path.join(self.result_dir, f'dcm/{i+1}_SIEMENS_{str(c_org.numpy())}_to_GE_{c_ge.cpu()}.dcm')
                                self.save_dicom(path, predict, dcm_save_path)
                            elif num == 1:
                                dcm_save_path = os.path.join(self.result_dir, f'dcm/{i+1}_GE_{str(c_org.numpy())}_to_GE_{c_ge.cpu()}.dcm')
                                self.save_dicom(path, predict, dcm_save_path)
                            print(f'Saved fake dicom image into {dcm_save_path}...')

                    # Save the translated images.
                    x_concat = torch.cat(x_fake_list, dim=3)
                    if num == 0:
                        png_save_path = os.path.join(self.result_dir, f'png/{i+1}_SIEMENS_{str(c_org.numpy())}.png')
                        save_image(self.denorm(x_concat.data.cpu()), png_save_path, nrow=1, padding=0)
                    elif num == 1:
                        png_save_path = os.path.join(self.result_dir, f'png/{i+1}_GE_{str(c_org.numpy())}.png')
                        save_image(self.denorm(x_concat.data.cpu()), png_save_path, nrow=1, padding=0)
                    print(f'Saved real and fake images into {png_save_path}...')