import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

NUM_FRAMES      = 16
IMAGE_SIZE      = 224
UCF101_ROOT     = '/mnt/data1/lzy/vivit/UCF-101'
ANNOTATION_PATH = '/mnt/data1/lzy/vivit/ucfTrainTestlist'


class UCF101Dataset(Dataset):
    def __init__(self, root, annotation_path, num_frames, is_train, fold=1):
        self.num_frames = num_frames
        self.is_train   = is_train
        self.dataset    = torchvision.datasets.UCF101(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=self.num_frames * 2 if is_train else self.num_frames,
            step_between_clips=self.num_frames//2 if is_train else 4,
            train=is_train,
            fold=fold,
            output_format="TCHW",
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, audio, label = self.dataset[idx]
        T = video.shape[0]
        if self.is_train:
            start = torch.randint(0, T - self.num_frames + 1, ()).item()
        else:
            start = (T - self.num_frames) // 2
        video = video[start : start + self.num_frames]
        return video, label


class VideoTransformGPU(nn.Module):
    def __init__(self, is_train: bool, image_size: int = IMAGE_SIZE):
        super().__init__()
        self.is_train   = is_train
        self.image_size = image_size
        self.resize     = transforms.Resize((256, 256), antialias=True)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # 输入：(B, T, C, H, W) uint8
        B, T, C, H, W = video.shape

        video = video.float() / 255.0
        video = video.reshape(B * T, C, H, W)
        video = self.resize(video)

        if self.is_train:
            # 同一视频所有帧共享同一组裁剪参数，保证时序一致性
            video = video.reshape(B, T, C, 256, 256)
            i, j, h, w = transforms.RandomCrop.get_params(
                video[:, 0], output_size=(self.image_size, self.image_size)
            )
            video = video.reshape(B * T, C, 256, 256)
            video = TF.crop(video, i, j, h, w)
            if torch.rand(1).item() > 0.5:
                video = TF.hflip(video)
        else:
            video = TF.center_crop(video, self.image_size)

        video = (video - self.mean) / self.std
        video = video.reshape(B, T, C, self.image_size, self.image_size)
        video = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return video


def get_dataloaders(batch_size, num_workers=4, is_distributed=False):
    train_dataset = UCF101Dataset(UCF101_ROOT, ANNOTATION_PATH, NUM_FRAMES, is_train=True,  fold=1)
    test_dataset  = UCF101Dataset(UCF101_ROOT, ANNOTATION_PATH, NUM_FRAMES, is_train=False, fold=1)

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        test_sampler  = SequentialSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        multiprocessing_context="forkserver",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        multiprocessing_context="forkserver",
    )
    return train_loader, test_loader
