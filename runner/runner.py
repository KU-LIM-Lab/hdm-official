import os
import logging
import glob
import time
import tqdm
from cleanfid import fid

from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

import torchvision 
import torchvision.utils as tvu

from models.ddpm import Model
from models.fno import FNO2
from models.unet_mnist import Unet

from functions.utils import *
from functions.loss import loss_fn
from functions.sde import VPSDE
from functions.ihdm import IHDM
from functions.sampler import Sampler

from evaluate.prdc import calculate_given_paths

from transformers import get_scheduler
from datasets import data_scaler, data_inverse_scaler

torch.autograd.set_detect_anomaly(True)

class Diffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config
        
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps

        self.sde = VPSDE(config.diffusion.beta_schedule, k=config.data.image_size, 
                         index = config.diffusion.index, sig=config.diffusion.sig)

        self.dataset = dataset
        self.test_dataset = test_dataset

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if args.distributed:
            sampler = DistributedSampler(self.dataset, shuffle=True, 
                                     seed=args.seed if args.seed is not None else 0)
            eval_sampler = DistributedSampler(self.testd_dataset, shuffle=False,
                                     seed=args.seed if args.seed is not None else 0)
        else:
            sampler = None
            eval_sampler = None
        
        train_loader = data.DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            sampler=sampler
        ) 

        val_loader = data.DataLoader(
            self.test_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            sampler=eval_sampler)
        
        # Model 
        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional)
        elif config.model.model_type == "FNO":
            model = FNO2(config)
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)
        
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                                                            device_ids=[args.local_rank],)
        logging.info("Model loaded.")

        # Optimizer, LR scheduler 
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr)
        
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=config.training.n_iters
        )

        start_epoch, step, val_step = 0, 0, 0
        if args.resume:
            states = torch.load(os.path.join(args.log_path, "ckpt.pth"), map_location=self.device)
            model.load_state_dict(states[0], strict=False)
            start_epoch = states[2]
            step = states[3]
            try:
                lr_scheduler.load_state_dict(states[4])
            except:
                pass
        
        for epoch in range(start_epoch, config.training.n_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            data_start = time.time()
            data_time = 0

            model.train()
            for i, batch in enumerate(train_loader):
                x = batch[0]
                n = x.size(0)

                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_scaler(x)
                
                t = torch.rand(n, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                e = torch.randn(x.shape, device=x.device)
                
                loss = loss_fn(model, self.sde, x,t, e).to(self.device)
                tb_logger.add_scalar("train_loss", torch.abs(loss), global_step=step)

                optimizer.zero_grad()
                loss.backward()

                if args.local_rank == 0:
                    logging.info(
                        f"step: {step}, loss: {torch.abs(loss).item()}, data time: {data_time / (i+1)}"
                    )

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
            
                optimizer.step()
                lr_scheduler.step()

                if step % config.training.ckpt_store == 0:
                    states = [
                        model.module.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                        lr_scheduler.state_dict()
                    ]

                    torch.save(
                        states,
                        os.path.join(args.log_path, "ckpt_{}.pth".format(step)),
                    )

                    self.ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(states, self.ckpt_dir)
            
                data_start = time.time()

            if config.training.validation_freq % step == 0:
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        x = batch[0]
                        n = x.size(0)

                        data_time += time.time() - data_start
                        val_step += 1

                        x = x.to(self.device)
                        x = data_scaler(x)
                        
                        t = torch.rand(n, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                        e = torch.randn(x.shape, device=x.device)
                        
                        loss = loss_fn(model, self.sde, x,t, e).to(self.device)
                        tb_logger.add_scalar("validation_loss", torch.abs(loss), global_step=val_step)

                        if args.local_rank == 0:
                            logging.info(
                                f"val step: {val_step}, loss: {torch.abs(loss).item()}, data time: {data_time / (i+1)}"
                            )
 
    def sample(self, score_model=None):
        args, config = self.args, self.config

        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional,)
        elif config.model.model_type == "FNO":
            model = FNO2(config)
        elif config.model.model_type == "ddpm":
            model = Model(config)
        
        model = model.to(self.device)

        if score_model is not None:
            model = score_model

        elif "ckpt_dir" in config.model.__dict__.keys():
            ckpt_dir = config.model.ckpt_dir
            states = torch.load(
                ckpt_dir,
                map_location=config.device,
            )

            if args.distributed:
                state_dict = OrderedDict()
                for k, v in states[0].items():
                    if 'module' in k:
                        name = k[7:]
                        state_dict[name] = v
                    else:
                        state_dict[k] = v

                model.load_state_dict(state_dict)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            else:
                model.load_state_dict(states[0], strict=False)
        else:
            raise Exception("Fail to load model due to invalid ckpt_dir")

        logging.info("Done loading model")
        model.eval()
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        logging.info(f"starting from image {img_id}")
        
        sampler = Sampler(args, config, model, device=self.device)
        
        if self.args.fid:
            indices = torch.arange(0, self.config.sampling.num_sample)
            dataset = data.Subset(self.dataset, indices)
                
            sample_loader = data.DataLoader(
                dataset,
                batch_size=self.config.sampling.batch_size,
                num_workers=self.config.data.num_workers,
            ) 

            total_n_samples = config.sampling.num_sample
            dataname = self.config.data.dataset
            dataset_path = "data/" + dataname.lower() + "_train_fid"

            try:
                dataset_path = str(dataset_path) + self.config.data.category
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                    data_store(sample_loader, dataset_path)
            except:
                pass

            n_gpus = dist.get_world_size()
            n_rounds = (total_n_samples - img_id) // self.config.sampling.batch_size // n_gpus # total num of datasamples (cifar10 has 5000 training dataset)
            remainder = (total_n_samples - img_id) // self.config.sampling.batch_size % n_gpus
            if dist.get_world_size() > 1 and remainder != 0:
                if args.local_rank < remainder:
                    n_rounds = n_rounds + 1

        else:
            n_rounds = 1

        if args.distributed:
            try:    
                sampler = DistributedSampler(self.test_dataset, shuffle=True, 
                                     seed=args.seed if args.seed is not None else 0)
            except:
                sampler = None
        else:
            sampler = None

        test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=config.sampling.batch_size,
            num_workers=config.data.num_workers,
            sampler=sampler
        )    

        if args.sample_type == 'sde_super_resolution':
            with torch.no_grad():
                for i, origin_img in tqdm(enumerate(test_loader)):
                    if origin_img[0][1] == 1:
                        img = origin_img[0].to(self.device).repeat_interleave(3, dim=1)
                    else:
                        img = origin_img[0].to(self.device)
                    
                    
                    low = config.data.image_size
                    multiplier = config.data.image_size // low
                    tvu.save_image(
                        origin_img[0], os.path.join(args.image_folder, f"{i}th_origin.png")
                    , dpi = 500)

                    if args.degraded_type == 'blur':
                        b = torchvision.transforms.GaussianBlur(kernel_size =(config.data.image_size -1,
                                                                config.data.image_size-1), sigma=(40,50))
                        img = b(img)
                    elif args.degraded_type == 'pixelate':
                        img = img.repeat_interleave(int(multiplier ),dim=-1).repeat_interleave(int(multiplier ),dim=-2)
                    else:
                        raise NotImplementedError(f'Unknown degraded type {args.degraded_type}')
                    
                    img = data_scaler(img)
                    
                    n  = len(img)
                    t = torch.ones(n, device=self.device) * self.sde.T
                    z = data_inverse_scaler(img)
            
                    tvu.save_image(z, os.path.join(args.image_folder, f"{i}th_before.png"))
                    
                    free_sde = VPSDE(config.diffusion.beta_schedule, k=config.data.image_size, 
                                     index = config.diffusion.index, sig=config.diffusion.sig)
                    x = free_sde.marginal_std(torch.randn(img.shape).to(self.device),t)
                    x = sampler(args, config, x, None, model, free_sde, img, device=self.device, multiplier=multiplier, name=f'{i}th')

                    y = data_inverse_scaler(x)
                    
                    tvu.save_image(y, os.path.join(args.image_folder, f"{i}th'result.png"),dpi =500  )
                    img = x 
                    img_id +=1
                    break

        elif args.sample_type == 'sde_imputation':                        
            with torch.no_grad():
                for i, batch in tqdm(enumerate(test_loader)):
            
                    img = batch[0].to(self.device)
                    img = data_scaler(img)

                    n  = len(img)
                    t = torch.ones(n, device=self.device) * self.sde.T
                    
                    tvu.save_image(
                        origin_img[0], os.path.join(args.image_folder, f"{i}_origin.png")
                    , dpi = 500)
                
                    mask = torch.ones_like(img).to(self.device)
            
                    mask = torch.randint(0,2, size=(int(config.data.image_size/4), int(config.data.image_size/4))).to(self.device)
                    mask = mask.repeat_interleave(int(4),dim=-1).repeat_interleave(int(4),dim=-2)
                                            
                    # To increaase the image size 
                    img = img * mask
                    z = data_inverse_scaler(img)
                    z[z == 0.5] = 1

                    tvu.save_image( z, os.path.join(args.image_folder, f"{i}_masked_image.png"))
                    
                    x = torch.randn_like(img)
                    free_sde = VPSDE(config.diffusion.beta_schedule, k=config.data.image_size, 
                                     index=config.diffusion.index, sig=config.diffusion.sig,)

                    x = sampler(args, config, x, None, model, free_sde, img, mask, device=self.device)
                    x = data_inverse_scaler(x)

                    tvu.save_image(x, os.path.join(args.image_folder, f"{i}_impainting.png"), dpi =500)
                    img_id +=1
            
        else:
            with torch.no_grad():
                for k in tqdm(
                    range(n_rounds), desc="Generating image samples"
                ):  
                    if args.local_rank == 0:
                        logging.info(
                            f"{k} / {n_rounds} sampling done"
                        )

                    n = config.sampling.batch_size

                    x_shape = (n, config.data.channels, config.data.image_size, config.data.image_size)
                   

                    y= None
                    if args.prior == 'ihdm':
                        x = IHDM.get_initial_samples(config, self.test_dataset, n)
                    else:
                        x = torch.randn(x_shape).to('cuda')
                      
                    S = Sampler(args, config, model, device='cuda')
                    x = S.sample(self.sde, x)

                    x = data_inverse_scaler(x)
                    x = x.clamp(0, 1.0)
                    
                    n = x.shape[0]

                    if args.fid:
                        dir_path = os.path.join(self.args.image_folder, 'sampling')
                        if os.path.isdir(dir_path) is False:
                            try:
                                os.mkdir(self.args.image_folder)
                            except:
                                pass

                        for i in range(n):
                            tvu.save_image(
                                x[i], os.path.join(args.image_folder, f"{args.local_rank}_{img_id}.png"),dpi =500
                            )
                            img_id += 1

                    else:
                        if args.local_rank == 0:    
                            name = str(2)+'_'+str(time.strftime('%m%d_%H%M_', time.localtime(time.time())))+'.png'
                            tvu.save_image(x, 'image.png')
                            try:
                                tvu.save_image(x, os.path.join(args.exp,'samples',name),nrow= int(np.sqrt(n)), dpi =500)
                            except:
                                os.mkdir(os.path.join(args.exp,'samples',name))
                                tvu.save_image(x, os.path.join(args.exp,'samples',name),nrow= int(np.sqrt(n)), dpi =500)


            if args.fid:
                if args.local_rank == 0:
                    fid_score = fid.compute_fid(fdir1=dataset_path, fdir2=args.image_folder, mode='clean')
                    prdc = calculate_given_paths(path1=dataset_path, path2=args.image_folder)
                    logging.info(f"FID score on {config.data.dataset} dataset with {config.sampling.num_sample} images: {fid_score:.3f}")
                    logging.info(f"Precision: {prdc[0]:.3f}, Recall: {prdc[1]:.3f}, Diversity: {prdc[2]:.3f}, Coverage: {prdc[3]:.3f}")
