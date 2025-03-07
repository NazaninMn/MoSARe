"""
Cancer Subtype Classification with Self-Normalizing Neural Networks

This script implements a 5-fold cross-validation pipeline for cancer subtype classification
using RNA-seq data and Self-Normalizing Neural Networks (SNNs).

Features:
- StandardScaler normalization
- Early stopping with ReduceLROnPlateau scheduling
- Multiple evaluation metrics (AUC, Accuracy, F1, Recall, Precision)
- Reproducible results through seed setting
- GPU acceleration support
- Comprehensive logging

Usage:
python main.py --data_root <split_files_dir> \
              --rna_path1 <luad_data.csv> \
              --rna_path2 <lusc_data.csv> \
              --batch_size 128 \
              --max_epochs 100 \
              --dropout 0.2
"""

import os
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, recall_score, confusion_matrix)
from tqdm import tqdm
from collections import defaultdict

# ####################
# Configuration Setup
# ####################

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cancer Subtype Classification with SNNs')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to cross-validation split files')
    parser.add_argument('--rna_path', type=str, required=True,
                       help='Path to processed RNA-seq data1 (CSV format)')
    parser.add_argument('--lusc_path', type=str, required=True,
                       help='Path to processed LRNA-seq data2 (CSV format)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Input batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='L2 regularization weight')
    return parser.parse_args()

# ################
# Model Definition
# ################

class SNN(nn.Module):
    """Self-Normalizing Neural Network for cancer subtype classification
    
    Architecture:
    - Input layer: 512 units
    - Hidden layer: 256 units
    - Output layer: 2 units
    - Activation: SELU with AlphaDropout
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output classes
        p_drop (float): Dropout probability
    """
    
    def __init__(self, in_features: int, out_features: int, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SELU(),
            nn.AlphaDropout(p_drop),
            
            nn.Linear(512, 256),
            nn.SELU(),
            nn.AlphaDropout(p_drop),
            
            nn.Linear(256, out_features)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using LeCun normal initialization"""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        """Forward pass"""
        return self.net(x)

# ###############
# Data Processing
# ###############

class DataHandler:
    """Handle data loading and preprocessing"""
    
    def __init__(self, args):
        self.args = args
        self._load_data()
    
    def _load_data(self):
        """Load and merge RNA-seq data"""
        # Load datasets
        luad = self._load_rna_data(self.args.rna_path, 0)
        lusc = self._load_rna_data(self.args.lusc_path, 1)
        
        # Merge and validate
        self.merged_data = pd.concat([luad, lusc]).reset_index(drop=True)
        if self.merged_data.isnull().values.any():
            raise ValueError("Missing values detected in merged data")
            
    @staticmethod
    def _load_rna_data(path, label):
        """Load single RNA-seq dataset"""
        df = pd.read_csv(path, index_col=0)
        df['patient_id'] = df.index.str[:12]  # Extract patient ID
        df['label'] = label
        return df
    
    @staticmethod
    def load_fold_split(data_root, fold_num):
        """Load cross-validation split"""
        train_path = os.path.join(data_root, f"train_fold_{fold_num}.csv")
        test_path = os.path.join(data_root, f"test_fold_{fold_num}.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Split files for fold {fold_num} not found")
            
        return (
            pd.read_csv(train_path)[['patient_id', 'label']].drop_duplicates(),
            pd.read_csv(test_path)[['patient_id', 'label']].drop_duplicates()
        )

# ##############
# Training Logic
# ##############

class Trainer:
    """Handle model training and evaluation"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.scaler = StandardScaler()
        self.data_handler = DataHandler(args)
        
    def prepare_dataloaders(self, fold):
        """Prepare DataLoaders for given fold"""
        # Load split
        train_fold, test_fold = DataHandler.load_fold_split(self.args.data_root, fold)
        
        # Merge with features
        train_data = self.data_handler.merged_data.merge(
            train_fold['patient_id'], on='patient_id')
        test_data = self.data_handler.merged_data.merge(
            test_fold['patient_id'], on='patient_id')
        
        # Preprocess
        X_train = self.scaler.fit_transform(
            train_data.drop(['label', 'patient_id'], axis=1))
        X_test = self.scaler.transform(
            test_data.drop(['label', 'patient_id'], axis=1))
        y_train = train_data['label'].values
        y_test = test_data['label'].values
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train))
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.LongTensor(y_test))
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args.batch_size)
        
        return train_loader, test_loader
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba[:, 1]),
            'f1': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'precision': tp / (tp + fp) if (tp := confusion_matrix(y_true, y_pred)[1,1]) + (fp := confusion_matrix(y_true, y_pred)[0,1]) > 0 else 0
        }
    
    def train(self, fold):
        """Full training loop for one fold"""
        # Prepare data
        train_loader, test_loader = self.prepare_dataloaders(fold)
        
        # Initialize model
        input_dim = train_loader.dataset.tensors[0].shape[1]
        model = SNN(input_dim, 2, self.args.dropout).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=self.args.patience//2)
        criterion = nn.CrossEntropyLoss()
        
        # Training state
        best_auc = 0
        best_model = None
        no_improve = 0
        
        # Training loop
        for epoch in range(self.args.max_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Evaluation phase
            model.eval()
            all_pred, all_proba, all_true = [], [], []
            val_loss = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = model(X)
                    val_loss += criterion(outputs, y).item()
                    proba = F.softmax(outputs, dim=1)
                    all_pred.extend(proba.argmax(1).cpu().numpy())
                    all_proba.extend(proba.cpu().numpy())
                    all_true.extend(y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            metrics = self.calculate_metrics(all_true, all_pred, np.array(all_proba))
            
            # Update scheduler
            scheduler.step(metrics['auc'])
            
            # Early stopping
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_model = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            
            # Logging
            print(f"Fold {fold} Epoch {epoch+1:03d}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"AUC: {metrics['auc']:.4f} (Best: {best_auc:.4f})")
            
            if no_improve >= self.args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model)
        return model

# ########
# Main Flow
# ########

def main():
    # Initial setup
    set_seed()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*40)
    print(f"Running 5-fold Cross Validation on {device}")
    print(f"RNA Data: {args.rna_path}")
    print(f"LUSC Data: {args.lusc_path}")
    print("="*40 + "\n")
    
    # Initialize components
    trainer = Trainer(args, device)
    final_metrics = defaultdict(list)
    
    # Cross-validation
    for fold in range(5):
        print(f"\n{'='*30} Fold {fold} {'='*30}")
        model = trainer.train(fold)
        
        # Final evaluation
        _, test_loader = trainer.prepare_dataloaders(fold)
        all_pred, all_proba, all_true = [], [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                outputs = model(X)
                proba = F.softmax(outputs, dim=1)
                all_pred.extend(proba.argmax(1).cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
                all_true.extend(y.numpy())
        
        metrics = trainer.calculate_metrics(all_true, all_pred, np.array(all_proba))
        for k, v in metrics.items():
            final_metrics[k].append(v)
        
        print(f"\nFold {fold} Results:")
        print(f"AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # Final report
    print("\n" + "="*40)
    print("Cross-Validation Results:")
    for metric, values in final_metrics.items():
        print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()