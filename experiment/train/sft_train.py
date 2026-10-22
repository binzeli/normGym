"""LoRA SFT of Qwen3-VL on the converted API-model traces.

Each sample is one turn: (system prompt, user prompt, step image) -> the JSON
response the API model produced. Loss is masked to the response tokens only.

Samples are processed one per device batch (variable image grids make cross-
sample padding fiddly and the corpus is small); use --grad_accum to set the
effective batch size.

Usage:
    python experiment/train/sft_train.py \
        [--dataset experiment/train/data/sft_train.jsonl] \
        [--epochs 2] [--lr 1e-4] [--out experiment/train/checkpoints/sft]
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader, Dataset

from experiment.train.config import (
    BASE_MODEL,
    SFT_DATASET,
    SFT_OUTPUT_DIR,
    SFT_VAL_DATASET,
)
from experiment.train.modeling import (
    build_messages,
    encode_full,
    load_model,
    load_processor,
    save_merged,
)


class SftDataset(Dataset):
    def __init__(self, jsonl_path: str, processor, max_len: int = 8192):
        self.processor = processor
        self.max_len = max_len
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = build_messages(
            sample['system'], sample['user'], sample['images']
        )
        encoded = encode_full(self.processor, messages, sample['response'])
        if encoded['input_ids'].shape[1] > self.max_len:
            # Overlong samples (huge memory sections) are truncated from the
            # left of the text is not safe with image tokens; drop by resample.
            return self[(index + 1) % len(self)]
        return encoded


def collate_single(batch):
    assert len(batch) == 1, 'per-device batch size is fixed at 1'
    return batch[0]


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 200) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        loss = model(**batch).loss
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_model', default=BASE_MODEL)
    parser.add_argument('--dataset', default=SFT_DATASET)
    parser.add_argument('--val_dataset', default=SFT_VAL_DATASET)
    parser.add_argument('--out', default=SFT_OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_len', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        # A silent CPU fallback OOMs the node (8B training needs a GPU); die
        # loudly instead — seen on the h100 partition after a broken driver
        # update (gh101, 2026-08-17).
        sys.exit('[FATAL] torch.cuda.is_available() is False on this node; '
                 'refusing to train on CPU. Check the partition/driver.')
    device = 'cuda'

    processor = load_processor(args.base_model)
    model = load_model(args.base_model, train_lora=True).to(device)
    model.train()
    model.config.use_cache = False
    # Full activations of an 8k-token multimodal sample overflow an 80GB card
    # (observed: 77GB in use at the first backward); checkpointing trades that
    # for recompute. enable_input_require_grads is required with LoRA so
    # gradients flow into the checkpointed blocks past the frozen embeddings.
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    train_data = SftDataset(args.dataset, processor, max_len=args.max_len)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,
                              collate_fn=collate_single)
    val_loader = None
    if args.val_dataset and os.path.exists(args.val_dataset):
        val_data = SftDataset(args.val_dataset, processor, max_len=args.max_len)
        if len(val_data):
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                                    collate_fn=collate_single)
    print(f'[OK] train samples: {len(train_data)}')

    steps_per_epoch = math.ceil(len(train_data) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    os.makedirs(args.out, exist_ok=True)
    step, running = 0, 0.0
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    print(f'[epoch {epoch} step {step}/{total_steps}] '
                          f'loss={running / args.log_every:.4f} '
                          f'lr={scheduler.get_last_lr()[0]:.2e}', flush=True)
                    running = 0.0

        adapter_dir = os.path.join(args.out, f'adapter_epoch{epoch + 1}')
        model.save_pretrained(adapter_dir)
        print(f'[OK] adapter saved: {adapter_dir}')
        if val_loader is not None:
            print(f'[epoch {epoch}] val loss: '
                  f'{evaluate(model, val_loader, device):.4f}', flush=True)

    # Final artifacts: the adapter (the RL trainer starts from this) and a merged
    # checkpoint (evaluable via run_experiment.py --provider hf).
    adapter_dir = os.path.join(args.out, 'adapter_final')
    model.save_pretrained(adapter_dir)
    merged_dir = save_merged(model, processor, os.path.join(args.out, 'merged'))
    print(f'[OK] final adapter: {adapter_dir}')
    print(f'[OK] merged checkpoint (for --provider hf eval): {merged_dir}')


if __name__ == '__main__':
    main()
