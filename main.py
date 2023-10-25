import argparse
import logging
import yaml
import os

import torch
import numpy as np
import torch.utils.tensorboard as tb

from runner import Diffusion, HilbertDiffusion
from datasets import get_dataset

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=False,
        default="",
        help="A string for documentation purpose."
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "--fid",
        default=None,
        choices=['train', 'test', None],
        help="type of fid calculation",
    )
    parser.add_argument(
        "--sample_type",
        default='sde',
        choices=['sde', 'ode', 'sde_imputation', 'sde_super_resolution' ,None],
        help="sampling method",
    )
    parser.add_argument(
        "--nfe", type=int, default=1000, help="number of function evaluations"
    )
    parser.add_argument(
        "--prior",
        default='hdm',
        choices=['hdm', 'ihdm'],
        help='Prior for sampling'
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )

    parser.add_argument(
        "--degrade_type",
        default='blur',
        choices=['blur', 'pixelate'],
        help='Type of generating degraded images'
    )

    parser.add_argument(
        "--distributed",
        action='store_true',
        help='Whether to use distributed data parallel or not'
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    args.image_folder = os.path.join( args.exp, "samples", args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    if args.distributed and torch.distributed.is_available():
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        args.local_rank = 0

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if args.local_rank == 0:
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.sample:
        if not args.resume:
            if os.path.exists(args.log_path):
                pass
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not args.fid:
                    overwrite = False

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    dataset, test_dataset = get_dataset(config)

    if config.data.modality == '1D':
        runner = HilbertDiffusion(args, config, dataset, test_dataset)
    else:
        runner = Diffusion(args, config, dataset, test_dataset)

    if args.sample:
        runner.sample()
    else:
        runner.train()
        runner.sample()

if __name__ == "__main__":
    main()
