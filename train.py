import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from data_loader import VideoTransformGPU, get_dataloaders

# ==================== 全局配置 ====================
EPOCH                      = 40
LR                         = 1e-4
WEIGHT_DECAY               = 0.05
LABEL_SMOOTHING            = 0.1
BATCH_SIZE                 = 4
GRADIENT_ACCUMULATION_STEPS = 2
SAVE_PATH                  = "./vivit_ucf101_best.pth"


# ==================== 数据增强 ====================

def mixup_data(x, y, alpha=0.8):
    lam        = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index      = torch.randperm(batch_size).to(x.device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    B, C, T, H, W = x.size()
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(B).to(x.device)

    cx    = np.random.randint(W)
    cy    = np.random.randint(H)
    cut_w = int(W * math.sqrt(1 - lam))
    cut_h = int(H * math.sqrt(1 - lam))

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    x[:, :, :, y1:y2, x1:x2] = x[index][:, :, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== 学习率调度 ====================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ==================== 优化器 ====================

def build_optimizer(model, lr, weight_decay):
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr)


# ==================== 训练函数 ====================

def train_one_epoch(epoch, model, loader, criterion, optimizer,
                    scheduler, scaler, device, transform_gpu):
    model.train()
    optimizer.zero_grad()
    total_loss    = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}")
    for step, (videos, labels) in enumerate(pbar):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        videos = transform_gpu(videos)

        with autocast("cuda"):
            if torch.rand(1).item() < 0.5:
                videos, y_a, y_b, lam = mixup_data(videos, labels)
            else:
                videos, y_a, y_b, lam = cutmix_data(videos, labels)
            logits = model(videos)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss   = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss    += loss.item() * GRADIENT_ACCUMULATION_STEPS * videos.size(0)
        total_samples += videos.size(0)
        pbar.set_postfix(loss=f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")

    return total_loss / total_samples


# ==================== 评估函数 ====================

@torch.no_grad()
def evaluate(model, loader, criterion, device, transform_gpu, is_distributed=False):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for videos, labels in tqdm(loader, desc="Eval", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        videos = transform_gpu(videos)

        with autocast("cuda"):
            logits = model(videos)
            loss   = criterion(logits, labels)

        _, preds   = logits.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    # ✅ all_reduce 在循环外，对所有 batch 汇总后再做一次跨卡求和
    if is_distributed:
        stats = torch.tensor(
            [total_correct, total_loss, total_samples],
            dtype=torch.float64, device=device
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_correct, total_loss, total_samples = stats.tolist()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# ==================== Checkpoint ====================

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_acc):
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":   scaler.state_dict(),
        "best_acc":            best_acc,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["best_acc"]


# ==================== 绘图 ====================

def plot_curve(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["test_loss"],  "r-", label="Test Loss",  linewidth=2)
    ax1.set_title("Train Loss VS Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["test_acc"], "g-", label="Test Accuracy", linewidth=2)
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("./curve_figure.png", dpi=150)
    plt.close()


# ==================== 主函数 ====================

def main():
    # 初始化分布式环境
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = (local_rank == 0)

    # 数据加载
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4,
        is_distributed=True,
    )

    # GPU 预处理模块
    train_transform = VideoTransformGPU(is_train=True).to(device)
    test_transform  = VideoTransformGPU(is_train=False).to(device)

    # 模型
    from model import ViViT_Factorised_Encoder
    from config import VIVIT_BASE_UCF101 as config
    model = ViViT_Factorised_Encoder(config).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 损失、优化器、scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model, LR, WEIGHT_DECAY)
    scaler    = GradScaler("cuda")

    # 学习率调度器
    steps_per_epoch    = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps   = int(EPOCH * 0.2) * steps_per_epoch
    num_training_steps = EPOCH * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # 断点续训
    start_epoch = 0
    best_acc    = 0.0
    if is_main and os.path.exists(SAVE_PATH):
        print("发现 checkpoint，从断点续训")
        start_epoch, best_acc = load_checkpoint(
            SAVE_PATH, model, optimizer, scheduler, scaler, device
        )

    # 训练循环
    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in range(start_epoch, EPOCH):
        train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion,
            optimizer, scheduler, scaler, device, train_transform
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, test_transform,
            is_distributed=True
        )

        if is_main:
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(SAVE_PATH, epoch, model, optimizer, scheduler, scaler, best_acc)
                print(f"保存最优模型，Test Acc: {best_acc:.2f}%")

    if is_main:
        plot_curve(history)
        print(f"训练完成，最优 Test Acc: {best_acc:.2f}%")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
