# Graph Attention Networks (GAT) for Node Classification

This repository contains the implementation of the **Graph Attention Network (GAT)**, designed for node classification tasks on graph-structured data. The project replicates the experiments from the paper ["Graph Attention Networks"](https://arxiv.org/abs/1710.10903) by Veličković et al. The implementation supports the **Cora**, **Citeseer**, and **Pubmed** citation network datasets, providing a reproducible pipeline for training, evaluation, and visualization.

## Features
- Full implementation of GAT with multi-head attention and LeakyReLU-based self-attention.
- Automated dataset downloading and preprocessing embedded within the script.
- Training, validation, and evaluation scripts with performance visualizations.
- Easily configurable for hyperparameter tuning.

## Requirements
Ensure you have **Python 3.8+** installed. Use the following command to install the required libraries: `pip install -r requirements.txt`. The libraries required include:
- `torch` (PyTorch framework)
- `numpy`
- `scipy`
- `matplotlib`

## Running the Code
The dataset preprocessing, training, and evaluation steps are integrated into a single script. To run the entire pipeline, use the following commands:

### Training the Model
To train the GAT model for a specific dataset, use:
- For Cora: `python gat_transductive.py --dataset cora --epochs 1000 --lr 0.005`
- For Citeseer: `python gat_transductive.py --dataset citeseer --epochs 1000 --lr 0.005`
- For Pubmed: `python gat_transductive.py --dataset pubmed --epochs 1000 --lr 0.005`

### Evaluating the Model
The model evaluation happens automatically after training, and performance metrics are saved as plots. No additional command is required for evaluation.

## Arguments
| Argument         | Default | Description                               |
|------------------|---------|-------------------------------------------|
| `--dataset`      | cora    | Dataset to use (cora, citeseer, pubmed)   |
| `--epochs`       | 1000    | Number of training epochs                 |
| `--lr`           | 0.005   | Learning rate                             |
| `--dropout`      | 0.6     | Dropout rate                              |
| `--weight_decay` | 5e-4    | Weight decay for Adam optimizer           |
| `--hidden_units` | 8       | Number of hidden units per attention head |
| `--heads`        | 8       | Number of attention heads                 |
| `--alpha`        | 0.2     | Negative slope for LeakyReLU              |
| `--patience`     | 100     | Early stopping patience                   |

## How to Run the Project

### Step-by-Step Instructions
1. Clone the repository: `git clone https://github.com/mehulgoenka/CSE6250_Project.git` and navigate to the directory: `cd Code_v0.1`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the training pipeline: `python gat_transductive.py --dataset cora --epochs 1000`.
4. Visualize results: Open the generated `.png` files in the project directory to view accuracy and loss trends.

## Notes
Ensure you have `matplotlib` installed for generating plots. Results may vary slightly based on random initialization and hardware configuration.

## References
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. arXiv preprint arXiv:1710.10903.
