import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from opentad.models.bricks.conv import IF, RepeatTemporalDim, RemoveTemporalDim
import spikingjelly.clock_driven.neuron as neuron

import trans_utils

def replace_spikingnorm_by_ifnode(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_spikingnorm_by_ifnode(module)
        if module.__class__.__name__ == "SpikingNorm":
            model._modules[name] = neuron.IFNode(v_threshold=module.calc_v_th().data.item(),v_reset=None)
    return model


def set_model_timesteps(model, T):
    """Set SNN timesteps T on all IF neurons and ActionFormer modules.

    This replaces the need to manually modify T in conv.py and actionformer.py
    when switching between ANN (T=0) and SNN (T=8) modes.
    """
    for module in model.modules():
        if isinstance(module, IF):
            module.T = T
            module.expand.T = T
            module.merge.T = T
        if hasattr(module, "T") and module.__class__.__name__ == "ActionFormer":
            module.T = T


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["ann", "snn", "ann2snn"],
        help="test mode: 'ann' for ANN (T=0), 'ann2snn' for ANN (T=0), 'snn' for SNN (T=8)",
    )
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--monitor", default=True, type=bool)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to skip eval, only do inference")
    parser.add_argument("--T", type=int, default=None, help="SNN timesteps (default: 0 for ann, 8 for snn)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Determine T based on mode (can be overridden by --T)
    if args.T is not None:
        T = args.T
    else:
        T = 8 if args.mode == "snn" else 0

    # load config
    cfg = Config.fromfile(args.config)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    if args.rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")
    logger.info(f"Test mode: {args.mode}, T={T}")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model and set timesteps
    model = build_detector(cfg.model)
    set_model_timesteps(model, T)

    # DDP
    model = model.to(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # load checkpoint
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
    else:
        checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
    logger.info("Loading checkpoint from: {}".format(checkpoint_path))
    device = f"cuda:{args.rank % torch.cuda.device_count()}"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.mode == "ann":
        # ANN mode: load EMA or standard state_dict
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            model.load_state_dict(checkpoint["state_dict_ema"], strict=False)
            logger.info("Using Model EMA...")
        else:
            model.load_state_dict(checkpoint["state_dict"])
    elif args.mode == "ann2nn":
        # ANN2SNN mode: load EMA or standard state_dict, then replace ANN to SNN.
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            model.load_state_dict(checkpoint["state_dict_ema"], strict=False)
            logger.info("Using Model EMA...")
        else:
            model.load_state_dict(checkpoint["state_dict"])
        model = replace_spikingnorm_by_ifnode(model)
        model = trans_utils.replace_test_by_testneuron(model,0.99)
    else:
        # SNN mode: load model weights and apply SNN transformations
        model.load_state_dict(checkpoint["model"], strict=False)
        model = trans_utils.replace_test_by_testneuron(model)
        model.load_state_dict(checkpoint["model"], strict=False)
        model = trans_utils.replace_nonlinear_by_neuron(model)
        model = trans_utils.replace_at_by_neuron(model)
        model = trans_utils.replace_testneuron_by_twosideneuron(model, args)
        model = model.half()

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        args,
        args.rank,
        model_ema=None,
        use_amp=use_amp,
        world_size=args.world_size,
        not_eval=args.not_eval,
    )
    logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()
