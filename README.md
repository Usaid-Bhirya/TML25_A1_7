# Advanced Membership Inference Attack

This repository contains an implementation of an advanced membership inference attack against deep neural networks, specifically targeting ResNet18 models trained on image classification tasks.

## Overview

Membership inference attacks aim to determine whether a specific data sample was used in the training set of a machine learning model. This implementation uses sophisticated feature extraction and ensemble learning techniques to achieve high attack accuracy.

## Features

- **Comprehensive Feature Extraction**: Combines multiple types of features including:
  - Advanced confidence-based features (entropy, temperature scaling, top-k analysis)
  - Gradient-based features (gradient norms, statistics)
  - Layer activation features from intermediate ResNet layers
  - Shadow model predictions

- **Shadow Model Training**: Trains multiple diverse shadow models with:
  - Different subset sampling strategies (random vs. class-balanced)
  - Varied hyperparameters (learning rates, epochs)
  - Data augmentation techniques
  - Advanced optimization (AdamW, CosineAnnealingLR, label smoothing)

- **Advanced Ensemble Attack**: Uses multiple machine learning models:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Machine (RBF kernel)
  - Multi-layer Perceptron (Neural Network)
  - Weighted ensemble prediction with confidence calibration

## Requirements

```bash
torch
torchvision
numpy
pandas
scikit-learn
requests
tqdm
```

## File Structure

```
├── Membership Inference Attack.ipynb    # Main notebook
├── 01_MIA.pt                           # Target model weights
├── pub.pt                              # Public dataset (with membership labels)
├── priv_out.pt                         # Private dataset (to be evaluated)
└── test.csv                            # Output predictions
```

## Usage

### Basic Usage

```python
# Initialize the attack
attack = AdvancedMembershipInferenceAttack()

# Run the complete attack pipeline
ids, scores = attack.run_advanced_attack()

# Save results
df = pd.DataFrame({"ids": ids, "score": scores})
df.to_csv("test.csv", index=None)
```

### Advanced Configuration

```python
# Custom model path and device
attack = AdvancedMembershipInferenceAttack(
    target_model_path="./custom_model.pt",
    device='cuda'
)

# Custom dataset paths
ids, scores = attack.run_advanced_attack(
    public_dataset_path="./custom_pub.pt",
    private_dataset_path="./custom_priv.pt"
)
```

## Architecture

### Target Model
- **Model**: ResNet18 with 44-class output layer
- **Input**: RGB images normalized with mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]

### Attack Pipeline

1. **Data Loading**: Load public (labeled) and private (unlabeled) datasets
2. **Feature Extraction**: Extract comprehensive features from target model
3. **Shadow Training**: Train multiple shadow models on public data subsets
4. **Shadow Features**: Extract features from shadow models
5. **Attack Training**: Train ensemble of attack classifiers on public data
6. **Prediction**: Generate membership scores for private data

### Feature Types

#### Confidence Features
- Maximum probability and logit values
- Shannon entropy at different temperature scales
- Top-k probability analysis
- True class probability and ranking
- Margin analysis (difference between top predictions)
- Prediction correctness indicators
- Cross-entropy loss values

#### Gradient Features
- Gradient norm with respect to input
- Gradient mean and standard deviation
- Computed per sample during backpropagation

#### Activation Features
- Statistics from intermediate ResNet layers (layer1-4)
- Global average pooling followed by statistical measures
- Mean, standard deviation, min, max of activations

#### Shadow Features
- Predictions from multiple independently trained shadow models
- Maximum probabilities, entropies, losses, true class probabilities

## Performance

The attack achieves:
- **AUC**: ~0.67 (Area Under the ROC Curve)
- **TPR@FPR=0.05**: ~0.148 (True Positive Rate at 5% False Positive Rate)

## Dataset Classes

### TaskDataset
Basic dataset class for image classification tasks.

```python
class TaskDataset(Dataset):
    def __init__(self, transform=None)
    def __getitem__(self, index)  # Returns (id, image, label)
    def __len__(self)
```

### MembershipDataset
Extended dataset class that includes membership labels for training attack models.

```python
class MembershipDataset(TaskDataset):
    def __getitem__(self, index)  # Returns (id, image, label, membership)
```

## Key Methods

### Feature Extraction
- `extract_comprehensive_features()`: Main feature extraction pipeline
- `compute_advanced_confidence_features()`: Confidence-based features
- `compute_gradient_features()`: Gradient-based features
- `compute_layer_activations()`: Intermediate layer features

### Shadow Model Training
- `train_shadow_models()`: Train multiple diverse shadow models
- `train_single_shadow_model()`: Train individual shadow model
- `get_balanced_subset()`: Create class-balanced training subsets

### Attack Model Training
- `train_advanced_attack_models()`: Train ensemble of attack classifiers
- `predict_membership_advanced()`: Generate final membership predictions

## Security and Ethics

**⚠️ Important Notice**: This code is intended for research and educational purposes only. Membership inference attacks can pose privacy risks when applied to real-world systems. Users should:

- Only use this code on data and models they own or have explicit permission to test
- Follow ethical guidelines and institutional policies
- Consider the privacy implications of membership inference attacks
- Use findings to improve model privacy rather than exploit vulnerabilities

## Research Context

This implementation is based on research in machine learning privacy, particularly:
- Membership inference attacks on deep neural networks
- Shadow model training techniques
- Advanced feature engineering for privacy attacks
- Ensemble methods for attack model training