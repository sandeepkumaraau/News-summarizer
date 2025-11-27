# News Summarizer: Pegasus Fine-Tuning

This project implements a complete pipeline for fine-tuning Google's **Pegasus** model on the **CNN/DailyMail** dataset for abstractive text summarization. It is built using the Hugging Face `transformers` library and PyTorch.

## Features

- **Automated Pipeline**: Handles data preprocessing, tokenization, training, and evaluation.
- **Metric Tracking**: Tracks loss and ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) per epoch.
- **Visualization**: Automatically generates training/validation loss and ROUGE score plots.
- **Modular Design**: Code is organized into reusable modules within the `src/` directory.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sandeepkumaraau/News-summarizer.git
   cd News-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note: This project requires `torch`, `transformers`, `evaluate`, `nltk`, and `matplotlib`.*

## Usage

### Training

To start the fine-tuning process, simply run the main training script:

```bash
python train.py
```

The script will:
1. Download and preprocess the dataset.
2. Fine-tune the Pegasus model.
3. Evaluate using ROUGE metrics after every epoch.
4. Save the final model and tokenizer to `outputs/models/final_model`.
5. Save performance plots to the `outputs/` directory.

### Configuration

Hyperparameters and model settings are defined in `src/config.py`. You can modify these values to adjust the training behavior:

- `MODEL_NAME`: The pretrained model checkpoint (default: `google/pegasus-cnn_dailymail`).
- `EPOCHS`: Number of training epochs.
- `LEARNING_RATE`: Learning rate for the Adafactor optimizer.
- `BATCH_SIZE`: Training and validation batch sizes.
- `DEVICE`: Automatically detects CUDA, MPS (Mac), or CPU.

## Project Structure


```text
News-summarizer/
├── outputs/               # To save trained models and plots
│   ├── models/
│   └── plots/
├── src/                   # The core logic of your application
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and constants
│   ├── data.py            # Data loading, preprocessing, and collators
│   ├── model.py           # Model initialization
│   ├── trainer.py         # Training and evaluation loops
│   └── utils.py           # Helper functions (cleaning, metrics, plotting)
├── train.py               # The main entry point script
├── requirements.txt       # List of dependencies
├── .gitignore             # Files to ignore 
└── README.md
```

## Outputs

After training is complete, check the `outputs/` directory for:
- **Saved Model**: A Hugging Face compatible model folder.
- **Metrics**: Plots visualizing the training loss and ROUGE score progression.









