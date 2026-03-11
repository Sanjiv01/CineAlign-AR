import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.optim as optim
import json
import numpy as np
import logging
import datetime
import time
import decord
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from diffusers.training_utils import compute_density_for_timestep_sampling
from importlib import reload
from collections import OrderedDict
from decord import VideoReader

from utils.utils import to_

from models.wanxgen import WanxgenMulshot

import random
import shutil, glob

__all__ = ['main']


def time2frame(time):
    return [(t-1)*4+1 for t in time]


class VideoFolder(Dataset):
    """
    Dataset reading multi-shot clips as a single video tensor [C, T_sum, H, W].
    """
    def __init__(
        self,
        list_file,
        seq_len=61216,
        downsample=(4, 16, 16),
        min_area=round((1280 * 720) * 0.8),
        max_area=round((1280 * 720) / 0.8),
        fps=16,
        seed=2024,
        cfg=None,
    ):
        assert min_area <= max_area
        assert seq_len >= min_area / (downsample[1] * downsample[2])
        self.list_file = list_file
        self.seq_len = seq_len
        self.downsample = downsample
        self.min_area = min_area
        self.max_area = max_area
        self.fps = fps
        self.seed = seed
        self.cfg = cfg

        self.shuffle = False
        with open(list_file) as f:
            self.items = json.load(f)
        if self.shuffle:
            random.shuffle(self.items)

    def __getitem__(self, index):
        while True:
            try:
                return self.get_item(index)
            except Exception as e:
                print(e)
                path = self.items[index]['shots']
                print(f'Error when loading {path}. Trying another one.')
                index = random.randint(0, len(self.items) - 1)
                continue

    def get_item(self, index):
        item = self.items[index]
        paths, caption = item['shots'], item['cap_list']
        inner_t = item['inner_t']  # list of per-shot frame counts in latent grid time (after downsample)

        # seeding (need to reseed for each thread)
        rng = np.random.default_rng(self.seed + hash(paths[0]) % 10000)

        # decord
        decord.bridge.set_bridge('torch')

        video_list = []
        for i, path in enumerate(paths):
            # print(path)
            # import pdb; pdb.set_trace()
            path = "/projectnb/cs523aw/students/erioe/Style-Aligned-Multi-Shot-Video-Generation/EchoShot/data/video/" + path
            # print(path)
            reader = VideoReader(path)
            fps = reader.get_avg_fps()
            frame_timestamps = np.array([
                reader.get_frame_timestamp(i) for i in range(len(reader))
            ], dtype=np.float32)
            duration = frame_timestamps[-1].mean()
            df, dh, dw = self.downsample

            of = inner_t[i]
            of = (of - 1) * df + 1  # map latent-time length back to frame count
            oh = self.cfg.H
            ow = self.cfg.W

            # sample frame ids uniformly over the target duration
            target_duration = of / self.fps
            begin = rng.uniform(0, max(1e-6, duration - target_duration))
            timestamps = np.linspace(begin, begin + target_duration, of)
            frame_ids = np.argmax(np.logical_and(
                timestamps[:, None] >= frame_timestamps[None, :, 0],
                timestamps[:, None] < frame_timestamps[None, :, 1]
            ), axis=1).tolist()

            video = reader.get_batch(frame_ids)             # [t, h, w, c]
            video = self._preprocess_video(video, oh, ow)   # [c, t, h, w]
            video_list.append(video)

        videos = torch.cat(video_list, dim=1)    # [c, t_sum, h, w]
        return videos, caption, inner_t

    def __len__(self):
        return len(self.items)

    def _preprocess_video(self, video, oh, ow):
        """Resize, center crop, convert to tensor, normalize to [-1, 1]."""
        # [t, h, w, c] -> [t, c, h, w]
        video = video.permute(0, 3, 1, 2)

        # resize + center crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # [t, c, h, w] -> [c, t, h, w] and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video


class BatchSampler(Sampler):
    """Simple infinite batch sampler."""
    def __init__(self, dataset_size, batch_size, seed=2024):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            yield self.rng.choice(
                self.dataset_size,
                self.batch_size,
                replace=self.dataset_size < self.batch_size
            )


def collate_fn(batch):
    videos, texts, inner_ts = zip(*batch)
    return list(videos), list(texts), list(inner_ts)


def _compute_seq_len_from_latents(z_list, patch_size):
    """Compute the required seq_len (max over batch) from latent sizes and patch_size."""
    pT, pH, pW = patch_size
    max_len = 0
    for z in z_list:  # each z: [C, Tz, Hz, Wz]
        _, tz, hz, wz = z.shape
        Ft = tz // pT
        Ht = hz // pH
        Wt = wz // pW
        max_len = max(max_len, Ft * Ht * Wt)
    return int(max_len)


def main(**kwargs):
    cfg.update(**kwargs)
    cfg.pmi_rank = int(os.environ['RANK'])
    cfg.pmi_world_size = int(os.environ['WORLD_SIZE'])
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size
    worker(cfg)


def worker(cfg):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    cfg.local_rank = local_rank
    print(f"[Rank {cfg.pmi_rank}] Using CUDA device {local_rank}")
    cfg.rank = cfg.pmi_rank
    gpu = cfg.rank % cfg.gpus_per_machine
    cfg.gpu = gpu
    print(f'rank: {cfg.pmi_rank} world_size: {cfg.pmi_world_size} gpus_per_machine: {cfg.gpus_per_machine}')
    
    # init distributed
    dist.init_process_group(
        backend='nccl',
        rank=cfg.rank,
        world_size=cfg.world_size,
        timeout=datetime.timedelta(hours=5)
    )

    # logging
    reload(logging)
    if cfg.rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=cfg.log_file),
                logging.StreamHandler(stream=sys.stdout)
            ]
        )
        logging.info(cfg)
    else:
        logging.basicConfig(level=logging.ERROR)

    # reduce noisy logs
    logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)
    from torch.distributed.checkpoint._dedup_tensors import logger
    logger.setLevel(logging.ERROR)

    # seeds
    cfg.seed += 1024 * cfg.rank
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator(device=gpu)
    g.manual_seed(cfg.seed)

    # dataloader
    logging.info('Initializing dataloader')
    dataset = VideoFolder(
        list_file=cfg.list_file,
        seq_len=cfg.seq_len,
        downsample=cfg.downsample,
        min_area=cfg.min_area,
        max_area=cfg.max_area,
        fps=cfg.fps,
        seed=cfg.seed,
        cfg=cfg,
    )
    sampler = BatchSampler(
        dataset_size=len(dataset),
        batch_size=cfg.batch_size,
        seed=cfg.seed
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory_device=f'cuda:{gpu}'
    )
    rank_iter = iter(dataloader)

    # init app & modules
    logging.info('Initializing T5, VAE, DiT, schedule, and diffusion process')

    # resume logic (optional)
    resume_step = 0
    if cfg.resume_train:
        parent_dir = cfg.log_dir
        logging.info(f'loading the latest checkpoint from logging folder: {parent_dir}')
        checkpoint_path = cfg.checkpoint_path
        if os.path.exists(parent_dir):
            checkpoints_dir = os.path.join(parent_dir, 'checkpoints')
            if os.path.exists(checkpoints_dir):
                step_dirs = glob.glob(os.path.join(checkpoints_dir, 'step_*'))
                max_step = -1
                for d in step_dirs:
                    try:
                        step = int(os.path.basename(d).split('_')[1])
                        max_step = max(max_step, step)
                    except Exception:
                        continue
                logging.info(f'latest step found: {max_step}')
                if max_step != -1:
                    step_dir = os.path.join(checkpoints_dir, f'step_{max_step}')
                    new_ckpt = os.path.join(step_dir, 'non_ema.pth')
                    if os.path.exists(new_ckpt):
                        checkpoint_path = new_ckpt
                        logging.info(f'using latest ckpt: {checkpoint_path} (override cfg.checkpoint_path)')
                        resume_step = max_step
        checkpoint_list = [checkpoint_path, None]
        dist.broadcast_object_list(checkpoint_list, src=0)
        cfg.checkpoint_path, cfg.checkpoint_ema_path = checkpoint_list

    app = WanxgenMulshot(
        device_id=gpu,
        fsdp_param_dtype=cfg.param_dtype if hasattr(cfg, 'param_dtype') else torch.bfloat16,
        fsdp_reduce_dtype=cfg.reduce_dtype,
        fsdp_buffer_dtype=cfg.buffer_dtype,
        fsdp_sharding_strategy=cfg.sharding_strategy,
        cfg=cfg
    )

    t5, vae, dit, dit_ema, schedule, diffusion = (
        app.t5, app.vae, app.dit, app.dit_ema, app.schedule, app.diffusion
    )

    # optimizer & scaler
    optimizer = optim.AdamW(
        params=dit.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = ShardedGradScaler(enabled=True, process_group=None)
    micro_steps = cfg.num_steps * cfg.grad_accum

    start = time.time()

    # training loop
    logging.info('Start the training loop')
    if resume_step != 0:
        logging.info(f'Skip data until step {resume_step}')

    for micro_step in range(1, micro_steps + 1):
        if micro_step <= resume_step:
            continue
        step = micro_step / cfg.grad_accum

        # read batch
        batch = next(rank_iter)
        batch = to_(batch, gpu, non_blocking=True)
        videos, texts, inner_t = batch

        # timesteps (no bias)
        u = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal',
            batch_size=len(videos),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        t = (u * schedule.config.num_train_timesteps).long()
        t = schedule.timesteps[t].to(device=videos[0].device)

        # preprocess
        with torch.no_grad():
            # encode videos to latents
            z = vae.encode(videos, inner_t=inner_t)

            # compute dynamic seq_len (max over batch) from latent shapes
            seq_len_dyn = _compute_seq_len_from_latents(z, patch_size=cfg.patch_size)

            # text context with CFG
            context = []
            null = t5([''])
            for u_text in texts:
                if rng.random() < cfg.p_zero:
                    context.append(null * len(u_text))
                else:
                    context.append(t5(u_text))

        # forward
        with amp.autocast(dtype=cfg.param_dtype):
            loss_list = diffusion.loss(
                x0=z,
                t=t,
                model=dit,
                model_kwargs={
                    'context': context,
                    'seq_len': seq_len_dyn,
                    'inner_t': inner_t
                },
                min_snr_gamma=cfg.min_snr_gamma,
                generator=g
            )
            # loss = torch.mean(torch.stack(loss_list))
            # unify + ensure scalar
            if isinstance(loss_list, torch.Tensor):
                loss = loss_list.mean()                          # <── 必須 mean
            elif isinstance(loss_list, (list, tuple)):
                loss = torch.stack(loss_list).mean()
            else:
                raise TypeError(f"Unexpected loss_list type: {type(loss_list)}")

        # backward
        scaler.scale(loss / cfg.grad_accum).backward()
        if micro_step % cfg.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA
            if cfg.use_ema and dit_ema is not None:
                with torch.no_grad():
                    ema_sd = OrderedDict(dit_ema.named_parameters())
                    model_sd = OrderedDict(dit.named_parameters())
                    for k, v in ema_sd.items():
                        v.copy_(model_sd[k].data.lerp(v, cfg.ema_decay))
                    del ema_sd, model_sd

        # metrics & logging
        loss_for_logging = loss.detach().float()
        dist.all_reduce(loss_for_logging)
        loss_for_logging /= cfg.world_size

        if cfg.rank == 0 and (
            step == 1 or step % cfg.log_interval == 0 or step == cfg.num_steps
        ):
            batch_size = cfg.batch_size * cfg.world_size
            throughput = (86400 * micro_step * batch_size) / max(1e-6, (time.time() - start))
            logging.info(
                f'Step: {int(step)}/{cfg.num_steps} '
                f'Loss: {loss_for_logging.item():.4f} '
                f'lr: {optimizer.param_groups[0]["lr"]:.6f} '
                f'scale: {scaler.get_scale():.2f} '
                f'throughput: {round(throughput):d} samples/day'
            )

        # checkpointing
        if step == cfg.num_steps or step % cfg.val_interval == 0:
            try:
                os.makedirs(cfg.log_dir, exist_ok=True)
                _, _, free = shutil.disk_usage(cfg.log_dir)
                if free / (1024**3) <= 20:
                    logging.info(f'*** Skip saving: only {int(free/(1024**3))} GB free. ***')
                else:
                    checkpoint_dir = osp.join(cfg.log_dir, f'checkpoints/step_{int(step)}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    logging.info(f'Saving state dict to {checkpoint_dir}')

                    # non-EMA
                    with FSDP.state_dict_type(
                        dit,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                    ):
                        non_ema = dit.state_dict()
                        if cfg.rank == 0:
                            torch.save(non_ema, osp.join(checkpoint_dir, 'non_ema.pth'))
                        del non_ema

                    # EMA
                    if cfg.use_ema and dit_ema is not None:
                        with FSDP.state_dict_type(
                            dit_ema,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                        ):
                            ema = dit_ema.state_dict()
                            if cfg.rank == 0:
                                torch.save(ema, osp.join(checkpoint_dir, 'ema.pth'))
                            del ema

                    logging.info(f'Checkpoint saved. Free: {int(free/(1024**3))} GB')
            except Exception as e:
                logging.error(f'Checkpoint saving failed: {e}')

    # finalize
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    if cfg.rank == 0:
        logging.info('Training completed!')


if __name__ == '__main__':
    from config_train import cfg
    main(cfg=cfg)
