from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from cs336_alignment.sft.utils import tokenize_prompt_and_output


def collate_fn(samples: list[tuple[str, str]], tokenizer):
    prompts = [s[0] for s in samples]
    responses = [s[1] for s in samples]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)  # type: ignore


class SftJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.dataset = load_dataset("json", data_files=path, split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        sample = self.dataset[idx]
        return sample["prompt"], sample["response"]


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)

    return dtype


def sft_train_step(
    model,
    batch: dict[str, Tensor],
    grad_accum_steps: int = 1,
    scaler=None,
) -> float:
    model.train()

    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)
    response_mask = batch["response_mask"].to(model.device)  # (B, T)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # ====== forward ======
    if scaler is not None:
        # 混合精度
        with autocast("cuda", get_torch_dtype(config["training"]["amp_dtype"])):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits  # (B, T, V)
    else:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    # ====== 计算 per-token log-prob ======
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, T, V)
    per_token_log_prob = torch.gather(
        log_probs_vocab,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)  # (B, T)

    log_probs_masked = per_token_log_prob * response_mask.to(per_token_log_prob.dtype)

    denom = response_mask.sum().clamp_min(1.0)
    loss = -log_probs_masked.sum() / denom  # scalar
    loss_to_backprop = loss / grad_accum_steps

    # ====== backward + step ======
    if scaler is not None:
        scaler.scale(loss_to_backprop).backward()
    else:
        loss_to_backprop.backward()

    return loss.item()


with Path(__file__).parent.joinpath("config.yaml").open() as f:
    config = yaml.safe_load(f)

model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    dtype=config["model"]["dtype"],
    device_map=config["model"]["device_map"],
    attn_implementation=config["model"]["attn_implementation"],
)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    target_modules=config["lora"]["target_modules"],
    lora_dropout=0.05,
    bias=config["lora"]["bias"],
    task_type=config["lora"]["task_type"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_dataset = SftJsonlDataset(config["data"]["sft_jsonl"])
loader = DataLoader(
    sft_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    collate_fn=partial(collate_fn, tokenizer=tokenizer),
)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(
    params,
    lr=float(config["training"]["learning_rate"]),
    weight_decay=config["training"]["weight_decay"],
)

use_amp = config["training"]["use_amp"]
scaler = GradScaler() if use_amp else None

num_epochs = config["training"]["num_epochs"]
grad_accum_steps = config["training"]["grad_accum_steps"]

for epoch in range(num_epochs):
    running_loss = 0.0
    step = 0

    optimizer.zero_grad()
    for i, batch in tqdm(enumerate(loader)):
        loss = sft_train_step(
            model,
            batch,
            grad_accum_steps=grad_accum_steps,
            scaler=scaler,
        )
        running_loss += loss
        step += 1

        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, config["training"]["max_grad_norm"])

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        if (i + 1) % 50 == 0:
            print(f"epoch {epoch} step {i + 1}  loss={running_loss / step:.4f}")

    print(f"[epoch {epoch}] mean loss = {running_loss / step:.4f}")

model.save_pretrained(config["output"]["model_dir"])
