## MacOS Shenanigans
import ssl
import nltk

## To avoid SSL certificate verification issues on MacOS


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler, Adafactor
from evaluate import load

from src import Config, get_dataloaders, train_one_epoch, evaluate, plot_metrics


def main():
    # Setup

    # MacOS Shenanigans
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
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
        if rouge:
            rouge_scores["rouge1"].append(rouge["rouge1"])
            rouge_scores["rouge2"].append(rouge["rouge2"])
            rouge_scores["rougeL"].append(rouge["rougeL"])
            print("ROUGE scores:", rouge)
        else:
            print("ROUGE scores not available for this epoch.")

        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"ROUGE: {rouge}")

    # Save & Plot
    model.save_pretrained("outputs/models/final_model")
    tokenizer.save_pretrained("outputs/models/final_model")
    plot_metrics(train_losses, val_losses, rouge_scores)


if __name__ == "__main__":
    main()
