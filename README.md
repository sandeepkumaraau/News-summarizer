# Fine-Tuning Pegasus for Abstractive Text Summarization 📝

This repository provides a complete pipeline to fine-tune and evaluate Google's state-of-the-art **Pegasus** model for abstractive text summarization. The project leverages the Hugging Face `Transformers` and `PyTorch` libraries to demonstrate the end-to-end process, from data preparation to model inference.

## Features

* **Data Processing**: Efficiently loads and preprocesses the `cnn_dailymail` dataset, including cleaning and tokenization.
* **Model Fine-Tuning**: A straightforward training script to fine-tune a pre-trained Pegasus checkpoint on your custom data.
* **Performance Evaluation**: Uses the standard **ROUGE** metric (ROUGE-1, ROUGE-2, ROUGE-L) to measure the quality of the generated summaries.
* **Inference Pipeline**: A simple interface to generate summaries for new, unseen text with your fine-tuned model.

---

