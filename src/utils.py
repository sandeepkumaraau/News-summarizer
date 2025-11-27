import re
import os
import nltk
import matplotlib.pyplot as plt


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects newline separated sentences
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def plot_metrics(
    train_losses, val_losses, rouge_scores=None, output_dir="outputs/plots"
):
    """
    Plots training metrics.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        rouge_scores (dict): Dictionary containing lists of ROUGE scores (e.g., {'rouge1': [...], 'rouge2': [...]}).
        output_dir (str): Directory to save plots.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.show()
    plt.close()

    # Plot ROUGE Scores
    if rouge_scores:
        plt.figure(figsize=(10, 5))
        for key, scores in rouge_scores.items():
            # Ensure we only plot if we have data for every epoch
            if len(scores) == len(epochs):
                plt.plot(epochs, scores, label=f"{key}")

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation ROUGE Scores")
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "rouge_plot.png"))
        plt.show()
        plt.close()
