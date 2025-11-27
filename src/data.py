from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from .utils import clean_text
from .config import Config


def pre_process(batch, tokenizer):
    inputs = [clean_text(article) for article in batch["article"]]
    targets = [clean_text(summary) for summary in batch["highlights"]]
    model_inputs = tokenizer(
        inputs, max_length=Config.MAX_INPUT_LEN, truncation=True, padding=False
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=Config.MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_dataloaders(tokenizer, model):
    """
    Creates Train and Validation DataLoaders.
    """

    raw_dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_raw = raw_dataset["train"]
    val_raw = raw_dataset["validation"]

    seed = getattr(Config, "SEED")

    train_shuffled = train_raw.shuffle(seed=seed)
    val_shuffled = val_raw.shuffle(seed=seed)

    train_size = int(len(train_shuffled) * Config.TRAIN_SUBSET_RATIO)
    val_size = int(len(val_shuffled) * Config.VAL_SUBSET_RATIO)

    train_subset = train_shuffled.select(range(train_size))
    val_subset = val_shuffled.select(range(val_size))

    tokenized_train = train_subset.map(
        lambda batch: pre_process(batch, tokenizer),
        batched=True,
        remove_columns=["article", "highlights", "id"],
    )
    tokenized_val = val_subset.map(
        lambda batch: pre_process(batch, tokenizer),
        batched=True,
        remove_columns=["article", "highlights", "id"],
    )

    # Initialize DataCollator once
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    train_loader = DataLoader(
        tokenized_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    return train_loader, val_loader
