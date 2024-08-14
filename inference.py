import argparse
import os
import sys
import time
import torch
from omegaconf import OmegaConf
import numpy as np
import torchvision
from pytorch_lightning import seed_everything
from sgm.util import instantiate_from_config, isheatmap
from torch.utils.data import  DataLoader
import torch.distributed as dist
import imageio
from PIL import Image
from einops import rearrange
MULTINODE_HACKS = True
from sgm.data.nuscenes_video.nuscenes_datasets_video import MyDataset
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import load_file as load_safetensors
import warnings
warnings.filterwarnings("ignore", category=Warning)
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--use_last_frame",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--split",
        type=str,
        const=True,
        default="train",
        nargs="?",
        help="split val or train",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=3407,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=4,
        help="batch size",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="use gpu",
    )
    parser.add_argument(
        "--inferdir",
        type=str,
        default="infers",
        help="directory for inference out",
    )
    parser.add_argument(
        "--ckptpath",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="postfix for logdir",
    )
    return parser

camera_views = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
viewid = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_BACK_RIGHT': 5,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_FRONT_LEFT': 2
}

def save_gif(tensor, filename):
    print("saving gif to",filename)
    tensor = (tensor + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    tensor = tensor.permute(0, 2, 3, 1)  # move color channel to the last
    tensor = (tensor * 255).byte()  # to 0-255 integer
    tensor_np = tensor.cpu().numpy()  # to numpy array
    if tensor_np.shape[-1] > 4:
        tensor_np = tensor_np[:, :, :, :10].min(-1)
    imageio.mimsave(filename, tensor_np, duration=1000/4,format="GIF", loop=0)  # save as gif
def logs_all_gifs(outs, root,filenames):
    images = outs
    for k in images:
        N = images[k].shape[0]
        if not isheatmap(images[k]):
            images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().float().cpu()
            images[k] = torch.clamp(images[k], -1.0, 1.0)
    for k in images:
        if 'txt' not in k and 'cond_img' not in k and 'reconstructions' not in k:
            all_images = rearrange(images[k], "(b t) c h w -> b t c h w ",t=8).contiguous()
            path_k = os.path.join(root, k)
            os.makedirs(path_k, exist_ok=True)
            for b in range(all_images.size(0)): 
                filename = filenames[-1][0][0].split('/')[-1].split('.')[0]+'.gif'
                path = os.path.join(path_k,filename)
                save_gif(all_images[b], path)
def logs_all_images(outs,root,filenames):
    for k in outs:
        N = outs[k].shape[0]
        if not isheatmap(outs[k]):
            outs[k] = outs[k][:N]
        if isinstance(outs[k], torch.Tensor):
            outs[k] = outs[k].detach().float().cpu()
            outs[k] = torch.clamp(outs[k], -1.0, 1.0)
    for k in outs:
        if 'cond_img' not in k and 'reconstructions' not in k:
            path_k = os.path.join(root,k)
            os.makedirs(path_k,exist_ok=True)
            grid = torchvision.utils.make_grid(outs[k], nrow=1)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            if grid.shape[-1] > 4:
                grid = grid[:,:,:10].min(-1)
            filename = "{}".format(
                filenames[-1][0][0].split('/')[-1].split('.')[0],
            )+'.png'
            path = os.path.join(path_k, filename)
            img = Image.fromarray(grid)
            img.save(path)


def logs_frames(jpgs,root,filenames):
    #jpgs 8 3 256 512*6
    N = jpgs.shape[0]
    if not isheatmap(jpgs):
        jpgs = jpgs[:N]
    if isinstance(jpgs, torch.Tensor):
        jpgs = jpgs.detach().float().cpu()
        jpgs = torch.clamp(jpgs, -1.0, 1.0)
    for view in camera_views:
        i = viewid[view]
        file_dir = filenames[-1][i][0].split('/')[-1].split('.')[0]
        path_view_vid = os.path.join(root, file_dir.split('__')[-2]+'_'+file_dir)
        os.makedirs(path_view_vid, exist_ok=True)
        for frame_id in range(8):
            grid = torchvision.utils.make_grid(jpgs[frame_id][:,:,512*i:512*i+512], nrow=1)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            if grid.shape[-1] > 3:
                grid = grid[:,:,:10].min(-1)
            filename = "_{:06}.jpg".format(frame_id)
            path = os.path.join(path_view_vid, filename)
            img = Image.fromarray(grid)
            img.save(path)

def model_load_ckpt(model, path):
    if path.endswith("ckpt"):
        if "deepspeed" in path:
            sd = torch.load(path, map_location="cpu")
            sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
        else:
            sd = torch.load(path, map_location="cpu")["state_dict"]
    elif path.endswith("safetensors"):
        sd = load_safetensors(path)
    else:
        raise NotImplementedError(f"Unknown checkpoint format: {path}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
    )
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected Keys: {unexpected}")

    return model
    
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    if not opt.name :
        raise ValueError(
            "You must specify the experiment name!!"
        )
    inferdir = os.path.join(opt.inferdir, opt.name)
    print(f"INFERENCE_DIR: {inferdir}")
    gifdir = os.path.join(inferdir,'gifs')
    allimage_dir = os.path.join(inferdir,'allimages')
    fake_dir = os.path.join(inferdir, 'fake')

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    seed = rank+opt.seed
    seed_everything(seed, workers=True)
    print('rank',rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # dataset
    split = opt.split
    use_last_frame = opt.use_last_frame
    print("build dataset:",split,'use last frame:',use_last_frame)
    print('seed:',seed)
    dataset = MyDataset(split=split,image_size=(512, 320), queue_length=8, render_pose=True,
                     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], bda_aug=False,use_last_frame=use_last_frame) 
    
    sampler = DistributedSampler(dataset, rank=rank,shuffle=False)
    mydataloader = DataLoader(
            dataset,
            batch_size=opt.bs,
            sampler=sampler,
        )
    
    # model
    model = instantiate_from_config(config.model) ## ckpt path in config
    print("load from:",opt.ckptpath)
    if opt.ckptpath is not None:
        model = model_load_ckpt(model,opt.ckptpath)
    else:
        print('warning!no newest checkpoint is loaded!')
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    print(config)
    gpu_autocast_kwargs = {
        "enabled": False, 
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    assert opt.bs == 1

    mydataloader = DataLoader(
            dataset,
            batch_size=opt.bs,
            sampler=sampler,
        )
    all_time = 0
    for idx,batch in enumerate(mydataloader):
        if idx % 10 == 0:
            print(f'idx {idx}, rank {rank}',opt.name,"len of dataloader", len(mydataloader))
        start = time.time()
        for key in batch:
            if key != 'txt' and key != 'filenames':
                batch[key]=batch[key].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outs = model.module.log_images(batch,sample_with_muldecoders=False)
        filenames = batch['filenames']
        logs_all_images(outs,allimage_dir,filenames)
        logs_all_gifs(outs, gifdir,filenames)
        logs_frames(outs['samples'], fake_dir,filenames)
        end = time.time()
        iter_time = end - start
        all_time += iter_time
        avg_iter_time = all_time / (idx + 1)
        if device.index == 0:
            if idx % 10 == 0:
                print("time per iter: %ss" % iter_time,
                      "avg time per iter: %ss" % avg_iter_time)
    print("save finished, device",device.index)
    time.sleep(6000)
