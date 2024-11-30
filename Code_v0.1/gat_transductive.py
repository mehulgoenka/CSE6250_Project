import os
import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from urllib.request import urlretrieve
import tarfile

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

######################
# Configuration
######################

# Define dataset URLs and file configurations
DATASET_CONFIG = {
    'cora': {
        'url': 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
        'content_file': 'cora.content',
        'cites_file': 'cora.cites',
        'num_features': 1433,
        'num_classes': 7
    },
    'citeseer': {
        'url': 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz',
        'content_file': 'citeseer.content',
        'cites_file': 'citeseer.cites',
        'num_features': 3703,
        'num_classes': 6
    },
    'pubmed': {
        'url': 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz',
        'content_file': 'Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab',
        'cites_file': 'Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab',
        'num_features': 500,
        'num_classes': 3
    }
}

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################
# Helper Functions
######################

def download_and_extract_dataset(dataset_name):
    """
    Downloads and extracts the specified dataset if not already present.

    Args:
        dataset_name (str): Name of the dataset ('cora', 'citeseer', 'pubmed').
    """
    config = DATASET_CONFIG.get(dataset_name)
    if config is None:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    dataset_dir = os.path.join(os.getcwd(), dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Define the path to the tar.gz file
    tar_path = os.path.join(dataset_dir, f"{dataset_name}.tgz")

    # Check if the required files already exist
    content_found = False
    cites_found = False
    for root, dirs, files in os.walk(dataset_dir):
        if config['content_file'] in files:
            content_found = True
        if config['cites_file'] in files:
            cites_found = True
        if content_found and cites_found:
            print(f"Dataset '{dataset_name}' already exists. Skipping download.")
            return

    # Download the dataset
    url = config['url']
    print(f'Downloading {dataset_name} dataset from {url}...')
    try:
        urlretrieve(url, tar_path)
        print('Download completed successfully.')
    except Exception as e:
        print(f"Failed to download {dataset_name} dataset. Error: {e}")
        sys.exit(1)

    # Extract the dataset
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar_ref, dataset_dir)
            print('Extraction completed successfully.')
    except Exception as e:
        print(f"Failed to extract {dataset_name} dataset. Error: {e}")
        sys.exit(1)

    # Remove the tar.gz file after extraction
    os.remove(tar_path)

def find_file_in_subdirectories(directory, filename):
    """
    Searches for a file within a directory and its subdirectories.

    Args:
        directory (str): The directory to search.
        filename (str): The name of the file to find.

    Returns:
        str: The full path to the file if found; None otherwise.
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def encode_onehot(labels):
    """
    One-hot encodes the labels.

    Args:
        labels (list or array): List of label strings.

    Returns:
        np.array: One-hot encoded labels.
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([classes_dict[label] for label in labels], dtype=np.int32)
    return labels_onehot

def normalize_features(features):
    """
    Row-normalizes the feature matrix.

    Args:
        features (scipy.sparse.csr_matrix): Feature matrix.

    Returns:
        scipy.sparse.csr_matrix: Normalized feature matrix.
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # Handle division by zero
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """
    Symmetrically normalizes the adjacency matrix.

    Args:
        adj (scipy.sparse.coo_matrix): Adjacency matrix.

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix.
    """
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # Handle division by zero
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Converts a scipy sparse matrix to a torch sparse tensor.

    Args:
        sparse_mx (scipy.sparse.coo_matrix): Sparse matrix.

    Returns:
        torch.sparse.FloatTensor: Torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    ).to(device)
    values = torch.from_numpy(sparse_mx.data).to(device)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)

def build_adjacency_matrix(cites_path, num_nodes, idx_map, dataset_name):
    """
    Builds a symmetric adjacency matrix from the citation file.

    Args:
        cites_path (str): Path to the citation file.
        num_nodes (int): Number of nodes in the graph.
        idx_map (dict): Mapping from paper IDs to indices.
        dataset_name (str): Name of the dataset.

    Returns:
        scipy.sparse.coo_matrix: Symmetrically normalized adjacency matrix.
    """
    # Load citation data
    edges_unordered = []
    if dataset_name == 'pubmed':
        # For Pubmed dataset
        with open(cites_path, 'r') as f:
            lines = f.readlines()
            # Skip header lines
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or 'NO_FEATURES' in line or 'DIRECTED' in line:
                    continue
                tokens = line.split('\t')
                if len(tokens) < 4:
                    continue  # Skip lines that don't have at least four tokens
                src_token = tokens[1]
                dst_token = tokens[3]
                # Remove 'paper:' prefix
                if src_token.startswith('paper:'):
                    src = src_token[len('paper:'):]
                else:
                    src = src_token
                if dst_token.startswith('paper:'):
                    dst = dst_token[len('paper:'):]
                else:
                    dst = dst_token
                if src in idx_map and dst in idx_map:
                    edges_unordered.append([idx_map[src], idx_map[dst]])
    else:
        # For Cora and Citeseer datasets
        edges_unordered = np.genfromtxt(cites_path, dtype=str)
        edges_unordered = np.array(
            [[idx_map[edge[0]], idx_map[edge[1]]] for edge in edges_unordered if edge[0] in idx_map and edge[1] in idx_map],
            dtype=np.int32
        )

    edges = np.array(edges_unordered, dtype=np.int32)
    if edges.size == 0:
        raise ValueError("No edges found in the citation data. Check if IDs in the citations file match those in the content file.")

    # Create adjacency matrix
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )

    # Make adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Add self-connections
    adj = adj + sp.eye(adj.shape[0])

    # Normalize adjacency matrix
    adj = normalize_adj(adj)

    return adj


def load_data(dataset_name='cora'):
    """
    Loads and preprocesses the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('cora', 'citeseer', 'pubmed').

    Returns:
        adj (torch.sparse.FloatTensor): Normalized adjacency matrix as a sparse tensor.
        features (torch.FloatTensor): Normalized feature matrix.
        labels (torch.LongTensor): Tensor of labels.
        idx_train (torch.LongTensor): Indices for training nodes.
        idx_val (torch.LongTensor): Indices for validation nodes.
        idx_test (torch.LongTensor): Indices for test nodes.
    """
    download_and_extract_dataset(dataset_name)

    config = DATASET_CONFIG.get(dataset_name)
    if config is None:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    print(f'Loading {dataset_name} dataset...')

    # Construct file paths
    dataset_dir = os.path.join(os.getcwd(), dataset_name)
    content_path = find_file_in_subdirectories(dataset_dir, os.path.basename(config['content_file']))
    cites_path = find_file_in_subdirectories(dataset_dir, os.path.basename(config['cites_file']))

    if content_path is None or cites_path is None:
        raise FileNotFoundError(f"Required files not found in {dataset_name} directory.")

    # Load features and labels
    if dataset_name == 'pubmed':
        # Special handling for Pubmed dataset
        idx_features_labels = []
        feature_names = set()
        with open(content_path, 'r') as f:
            lines = f.readlines()
            # Skip header lines starting with '#'
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                tokens = line.split('\t')
                if not tokens:
                    continue
                paper_id = tokens[0]
                label = None
                feature_dict = {}
                for token in tokens[1:]:
                    key_value = token.split('=')
                    if len(key_value) != 2:
                        continue
                    key, value = key_value
                    if key == 'label':
                        label = value
                    elif key.startswith('w-'):
                        # Collect feature names
                        feature_names.add(key)
                        feature_dict[key] = float(value)
                if label is None:
                    continue  # Skip if label is not found
                idx_features_labels.append((paper_id, feature_dict, label))
        if not idx_features_labels:
            raise ValueError("No data found in the content file.")

        # Create feature name to index mapping
        feature_name_to_index = {name: i for i, name in enumerate(sorted(feature_names))}
        num_features = len(feature_name_to_index)

        # Initialize feature matrix and labels
        features = sp.lil_matrix((len(idx_features_labels), num_features), dtype=np.float32)
        labels_list = []
        idx_list = []
        for idx, (paper_id, feature_dict, label) in enumerate(idx_features_labels):
            for key, value in feature_dict.items():
                feature_index = feature_name_to_index[key]
                features[idx, feature_index] = value
            labels_list.append(label)
            idx_list.append(paper_id)
        labels = encode_onehot(labels_list)
        idx = np.array(idx_list)
    else:
        # For Cora and Citeseer datasets
        idx_features_labels = np.genfromtxt(content_path, dtype=str)
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        idx = idx_features_labels[:, 0].astype(str)

    # Build index mapping
    idx_map = {j: i for i, j in enumerate(idx)}

    # Build adjacency matrix
    adj = build_adjacency_matrix(cites_path, num_nodes=labels.shape[0], idx_map=idx_map, dataset_name=dataset_name)

    # Normalize features
    features = normalize_features(features)

    # Convert to torch tensors
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Define train, validation, and test splits
    idx_train = []
    idx_val = []
    idx_test = []
    num_classes = labels.max().item() + 1

    if dataset_name == 'citeseer' or dataset_name == 'cora':
        # 20 nodes per class for training
        for i in range(num_classes):
            idx_i = (labels == i).nonzero(as_tuple=True)[0]
            idx_train.extend(idx_i[:20].tolist())
            idx_val.extend(idx_i[20:70].tolist())
            idx_test.extend(idx_i[70:].tolist())
    elif dataset_name == 'pubmed':
        # Pubmed dataset has more nodes; adjust splits accordingly
        for i in range(num_classes):
            idx_i = (labels == i).nonzero(as_tuple=True)[0]
            idx_train.extend(idx_i[:60].tolist())
            idx_val.extend(idx_i[60:560].tolist())
            idx_test.extend(idx_i[560:].tolist())

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    return adj, features, labels, idx_train, idx_val, idx_test


######################
# GAT Model Definitions
######################

def grouped_softmax(e, groups):
    """
    Compute softmax over groups.

    Args:
        e (torch.Tensor): Tensor of shape [E], attention coefficients.
        groups (torch.Tensor): Tensor of shape [E], group indices (source nodes).

    Returns:
        torch.Tensor: Attention coefficients after applying softmax per group.
    """
    # Groups are assumed to be sorted
    attention = torch.zeros_like(e)
    unique_groups, counts = torch.unique_consecutive(groups, return_counts=True)
    idx = 0
    for count in counts:
        e_group = e[idx:idx+count]
        e_max = torch.max(e_group)  # For numerical stability
        e_exp = torch.exp(e_group - e_max)
        e_exp_sum = torch.sum(e_exp)
        attention[idx:idx+count] = e_exp / e_exp_sum
        idx += count
    return attention

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer as described in the GAT paper.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        Initializes the graph attention layer.

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            dropout (float): Dropout probability.
            alpha (float): Negative slope for LeakyReLU activation.
            concat (bool): Whether to concatenate the multi-head attentions.
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # Input feature dimension
        self.out_features = out_features  # Output feature dimension
        self.alpha = alpha  # Negative slope for LeakyReLU
        self.concat = concat  # Whether to concatenate multi-head attentions

        # Linear transformation weight matrix
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Xavier initialization

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Activation functions
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        Forward pass for the GAT layer.

        Args:
            h (torch.FloatTensor): Input features, shape (N, in_features).
            adj (torch.sparse.FloatTensor): Normalized adjacency matrix.

        Returns:
            torch.FloatTensor: Output features.
        """
        Wh = torch.mm(h, self.W)  # Linear transformation

        # Extract edge indices
        edge_index = adj._indices()  # Shape: (2, E)

        # Node embeddings for edges
        Wh_i = Wh[edge_index[0]]  # Source node embeddings
        Wh_j = Wh[edge_index[1]]  # Target node embeddings

        # Concatenate embeddings and compute attention coefficients
        a_input = torch.cat([Wh_i, Wh_j], dim=1)  # Shape: (E, 2 * out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(1)  # Shape: (E,)

        # Sort edges by source node to group them
        src_nodes = edge_index[0]
        sorted_src_nodes, perm = torch.sort(src_nodes)
        e = e[perm]
        Wh_j = Wh_j[perm]
        src_nodes = sorted_src_nodes

        # Compute attention coefficients using grouped softmax
        attention = grouped_softmax(e, src_nodes)

        # Apply dropout to attention coefficients
        attention = self.dropout(attention)

        # Compute the new node features
        h_prime = torch.zeros_like(Wh)
        h_prime.index_add_(0, src_nodes, attention.unsqueeze(1) * Wh_j)

        if self.concat:
            # Apply activation function
            return F.elu(h_prime)
        else:
            # For the output layer
            return h_prime

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'

class GAT(nn.Module):
    """
    Graph Attention Network consisting of stacked GAT layers.
    """

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        Initializes the GAT model.

        Args:
            nfeat (int): Number of input features.
            nhid (int): Number of hidden units.
            nclass (int): Number of classes for prediction.
            dropout (float): Dropout probability.
            alpha (float): Negative slope for LeakyReLU activation.
            nheads (int): Number of attention heads.
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers (first layer)
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])

        # Output layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        """
        Forward pass of the GAT model.

        Args:
            x (torch.FloatTensor): Input features.
            adj (torch.sparse.FloatTensor): Normalized adjacency matrix.

        Returns:
            torch.FloatTensor: Log-softmax probabilities.
        """
        x = F.dropout(x, self.dropout, training=self.training)
        # Concatenate outputs from multiple attention heads
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # Output layer
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)

######################
# Training and Evaluation
######################

def accuracy(output, labels):
    """
    Computes the accuracy of predictions.

    Args:
        output (torch.Tensor): Model outputs (log probabilities).
        labels (torch.Tensor): True labels.

    Returns:
        float: Accuracy value.
    """
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)

def train_transductive(dataset_name='cora', num_epochs=1000, lr=0.005, weight_decay=5e-4,
                       nhid=8, nheads=8, dropout=0.6, alpha=0.2, patience=100):
    """
    Trains the GAT model on the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('cora', 'citeseer', 'pubmed').
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        nhid (int): Number of hidden units.
        nheads (int): Number of attention heads.
        dropout (float): Dropout rate.
        alpha (float): Negative slope for LeakyReLU.
        patience (int): Patience for early stopping.
    """
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_name)

    # Model and optimizer
    model = GAT(
        nfeat=features.shape[1],
        nhid=nhid,
        nclass=int(labels.max()) + 1,
        dropout=dropout,
        nheads=nheads,
        alpha=alpha
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping
    t_total = time.time()
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        # Compute loss and accuracy for training data
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        # Save losses and accuracies for plotting
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        train_accuracies.append(acc_train)
        val_accuracies.append(acc_val)

        print(
            f'Epoch: {epoch+1:04d}, '
            f'loss_train: {loss_train.item():.4f}, '
            f'acc_train: {acc_train:.4f}, '
            f'loss_val: {loss_val.item():.4f}, '
            f'acc_val: {acc_val:.4f}, '
            f'time: {time.time() - t:.4f}s'
        )

        # Early stopping
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{dataset_name}.pth')
        else:
            patience_counter += 1

        if patience_counter == patience:
            print('Early stopping!')
            break

    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total:.4f}s")

    # Load best model
    model.load_state_dict(torch.load(f'best_model_{dataset_name}.pth'))

    # Testing
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss ({dataset_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_plot_{dataset_name}.png')
    plt.show()

    # Plotting training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({dataset_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'accuracy_plot_{dataset_name}.png')
    plt.show()

######################
# Main Execution
######################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT Implementation for Transductive Datasets')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use: cora, citeseer, pubmed')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization, default: 5e-4)')
    parser.add_argument('--nhid', type=int, default=8, help='Number of hidden units (default: 8)')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (default: 0.6)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Negative slope for LeakyReLU (default: 0.2)')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience (default: 100)')
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    num_epochs = args.epochs
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose from cora, citeseer, pubmed.")

    train_transductive(
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        nhid=args.nhid,
        nheads=args.nheads,
        dropout=args.dropout,
        alpha=args.alpha,
        patience=args.patience
    )
