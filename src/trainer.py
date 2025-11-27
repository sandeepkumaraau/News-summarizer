import torch
from tqdm import tqdm
from .utils import postprocess_text
from .config import Config


def train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.autocast(device_type=Config.DEVICE.type, enabled=True):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Scheduler stepping every batch

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, loader, device, tokenizer, rouge_metric):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            generated_ids = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            decoded_preds = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            label_fixed = torch.where(
                batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]
            )
            decoded_labels = tokenizer.batch_decode(
                label_fixed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

    avg_loss = total_loss / num_batches
    rouge_result = rouge_metric.compute(predictions=all_preds, references=all_labels)
    return avg_loss, rouge_result
