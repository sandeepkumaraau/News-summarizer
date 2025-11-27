import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler, Adafactor
from evaluate import load

from src import Config, get_dataloaders, train_one_epoch, evaluate, plot_metrics


def main():
    # Setup
    nltk.download("punkt")
    print(f"Using device: {Config.DEVICE}")

    # Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE)

    # Data
    train_loader, val_loader = get_dataloaders(tokenizer, model)

    # Optimizer
    optimizer = Adafactor(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        scale_parameter=False,
        relative_step=False,
    )
    num_training_steps = Config.EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    rouge_metric = load("rouge")

    # Loop
    train_losses, val_losses = [], []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")

        t_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE
        )
        v_loss, rouge = evaluate(
            model, val_loader, Config.DEVICE, tokenizer, rouge_metric
        )

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Safely handle cases where evaluate() returned None or a non-dict result
        if isinstance(rouge, dict):
            rouge_scores["rouge1"].append(rouge.get("rouge1", 0.0))
            rouge_scores["rouge2"].append(rouge.get("rouge2", 0.0))
            rouge_scores["rougeL"].append(rouge.get("rougeL", 0.0))
        else:
            # append None placeholders (or use 0.0) to keep lists aligned with epochs
            rouge_scores["rouge1"].append(None)
            rouge_scores["rouge2"].append(None)
            rouge_scores["rougeL"].append(None)

        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"ROUGE: {rouge}")

    # Save & Plot
    model.save_pretrained("outputs/models/final_model")
    tokenizer.save_pretrained("outputs/models/final_model")
    plot_metrics(train_losses, val_losses, [])


if __name__ == "__main__":
    main()
