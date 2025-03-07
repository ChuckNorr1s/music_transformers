# A.T.C.T.P.F.M.G.C. - Audio Transformers Classification Training Pipeline For Music Genre Classification

This repository provides a production-ready pipeline for audio classification using Hugging Face's Transformers and Datasets libraries. The pipeline is divided into two main parts:

1. **Dataset Preparation**: Load audio data from a folder using the `audiofolder` loader and save it in Arrow format.
2. **Model Training**: Fine-tune a pretrained audio classification model on the prepared dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare the Dataset](#prepare-the-dataset)
  - [Train the Model](#train-the-model)
- [Configuration](#configuration)
- [Logging and Error Handling](#logging-and-error-handling)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The pipeline leverages the Hugging Face ecosystem to provide a robust training process for audio classification. It includes preprocessing steps (such as feature extraction with a specified sampling rate), label mapping, and fine-tuning with configurable training parameters. The code is modularized for clarity and production-readiness with extensive logging and error handling.

## Features

- **Modular Design:** Separate functions for data preparation, preprocessing, label mapping, and model training.
- **Command-line Interface:** Easily switch between dataset preparation and model training using subcommands.
- **Customizable Parameters:** Adjust hyperparameters (batch size, learning rate, epochs, etc.) and the number of processes (`--num_proc`) for dataset mapping via command-line arguments.
- **Robust Logging:** Detailed logging to monitor progress and debug issues.
- **Error Handling:** Graceful error management with appropriate exit codes.

## Requirements

- Python 3.10+
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [Evaluate](https://github.com/huggingface/evaluate)
- [Accelerate](https://github.com/huggingface/accelerate)
- [NumPy](https://numpy.org/)
- A CUDA-compatible GPU (optional, for fp16 training)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ChuckNorr1s/music_transformers.git
   cd music_transformers
   ```
2. **Create a Virtual Environment (optional but recommended):**
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

The script provides two subcommands: ```prepare``` and ```train```.

### Prepare the Dataset

To load your raw audio data from a folder and save it in Arrow format, run:

   ```bash
   python main.py prepare --data_dir ./dataset --output_dir ./music
   ```

- ```--data_dir:``` Directory containing your raw audio data.
- ```--output_dir:``` Directory where the prepared dataset will be saved.

### Train the Model

To fine-tune the pretrained audio classification model, run:

   ```bash
   python main.py train --dataset_disk ./music --model_id ntu-spml/distilhubert --num_train_epochs 10 --fp16 --num_proc 4
   ```

- ```--dataset_disk```: Path to the prepared dataset on disk.
- ```--model_id```: Pretrained model identifier from Hugging Face.
- ```--num_train_epochs```: Number of training epochs.
- ```--fp16```: Flag to enable mixed precision training.
- ```--num_proc```: Number of processes to use for dataset mapping (defaults to the number of CPUs).
- Additional parameters such as ```--batch_size```, ```--learning_rate```, and ```--experiment_name``` can also be configured.

## Configuration

The script supports various command-line arguments for both subcommands. Run:

```bash
python main.py -h
```
for detailed help on available commands and arguments.

## Logging and Error Handling

-    **Logging:** Critical steps (dataset loading, preprocessing, training) are logged. Check the console output for progress and error messages.
-    **Error Handling:** The script uses try/except blocks to catch and log exceptions. If an error occurs, an appropriate message is printed and the script exits gracefully.

## Troubleshooting

-    Dataset Loading Errors: Ensure your ```data_dir``` has the expected structure for the ```audiofolder``` loader.
-    Dependency Issues: Verify that all required libraries are installed and compatible with your Python version.
-    GPU/FP16 Issues: If you encounter issues with fp16 training, try running without the ```--fp16``` flag.
-    Process Configuration: Adjust the ```--num_proc``` parameter if you encounter performance issues during dataset mapping.

## License

[APACHE 2.0](https://choosealicense.com/licenses/apache-2.0/)
