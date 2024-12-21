# CIFAR-10 SVM Classifier

Welcome to the CIFAR-10 SVM Classifier repository. This project demonstrates the implementation of a Support Vector Machine (SVM) model to classify images from the CIFAR-10 dataset. The repository is designed to provide insights into machine learning workflows and best practices, catering to both research and production-ready implementations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Performance Metrics](#performance-metrics)
7. [Feature Engineering Details](#feature-engineering-details)
8. [File Mappings](#file-mappings)
9. [Future Enhancements](#future-enhancements)
10. [Contributing](#contributing)
11. [License](#license)

---

## Introduction

The CIFAR-10 dataset is a widely recognized benchmark for image classification tasks, containing 60,000 images across 10 distinct classes. This project leverages a Support Vector Machine (SVM) to classify the dataset, focusing on achieving robust performance while maintaining simplicity and interpretability.

---

## Key Features

- **Dataset Preprocessing**: Efficient handling of CIFAR-10 data, including normalization and feature extraction.
- **HOG + Color Histogram Feature Descriptor**: Combines Histogram of Oriented Gradients (HOG) and color histograms (implemented from scratch) for feature extraction.
- **PCA for Dimensionality Reduction**: Uses sklearn's PCA to retain 90% of total variance in the dataset.
- **t-SNE Visualization**: Includes 2D t-SNE plots for data visualization:
  - With PCA features.
  - With HOG + Color Histogram features.
- **GridSearchCV for SVM Optimization**: Utilizes GridSearchCV (cv=5) to find the best parameters (C, kernel, γ for Gaussian kernel):
  - For HOG + Color Histogram features.
  - For PCA features.
- **Support Vector-Based Training**: Implements a secondary training set derived from support vectors of the initial SVM, with comparison of accuracies:
  - For HOG + Color Histogram features.
  - For PCA features.
- **Ease of Use**: Clear modular code structure for easy customization.

---

## Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning Libraries**: scikit-learn, NumPy, pandas
- **Visualization Tools**: Matplotlib, seaborn

---

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/abhinavsaurabh/Cifar-10-SVM.git
cd Cifar-10-SVM
```

---

## Usage

### Training the Model

Run the following command to preprocess the data and train the SVM model:

```bash
python train.py
```

### Evaluate the Model

To evaluate the trained model, execute:

```bash
python evaluate.py
```

### Visualization

The repository includes scripts for visualizing data distribution and model performance:

```bash
python visualize.py
```

---

## Performance Metrics

The model achieves the following performance on the CIFAR-10 dataset:

- **Accuracy**: XX% (replace with actual value)
- **Precision, Recall, F1-Score**: Detailed metrics can be found in the evaluation logs.

---

## Feature Engineering Details

### HOG + Color Histogram

- Combines HOG and color histograms as a feature descriptor.
- Implemented from scratch for enhanced control and performance.

### PCA

- Performs Principal Component Analysis using sklearn.
- Retains 90% of the total variance in the dataset.

### t-SNE Visualizations

- **PCA t-SNE**: Visualizes the 2D t-SNE plot with PCA features.
- **HOG + Color Histogram t-SNE**: Visualizes the 2D t-SNE plot with combined HOG and color histogram features.

### GridSearchCV SVM Optimization

- **HOG + Color Histogram**: Uses GridSearchCV to optimize SVM hyperparameters (C, kernel, γ).
- **PCA**: Similar optimization process applied to PCA-reduced features.

### Support Vector-Based Training

- Develops a secondary training set from support vectors obtained in the initial SVM training.
- Compares accuracies between initial and secondary training for both feature types:
  - **HOG + Color Histogram**
  - **PCA**

---

## File Mappings

To facilitate navigation, the following file mappings correspond to specific features and tasks:

- **"1-1 HOG"**: `1-1_HOG.ipynb` - Combine HOG and color histogram (must be implemented from scratch).
- **"1-1 PCA"**: `1-1_PCA.ipynb` - Perform PCA using sklearn to retain 90% of total variance.
- **"1-2 PCA TSNE"**: `1-2_PCA_TSNE.ipynb` - Visualize the 2D t-SNE plot with PCA.
- **"1-2 TSNE_HOG+Color"**: `1-2_TSNE_HOG+Color.ipynb` - Visualize the 2D t-SNE plot with HOG and color histogram.
- **"1-3 HOG_rbr10_GridSearchCV scaled"**: `1-3_HOG_rbr10_GridSearchCV_scaled.ipynb` - Use GridSearchCV to find the best SVM parameters with HOG and color histogram.
- **"1-3 PCA rbf c = 10.py"**: `1-3_PCA_rbf_c=10.ipynb` - Use GridSearchCV to find the best SVM parameters with PCA.
- **"1-4 HOG"**: `1-4_HOG.ipynb` - Develop a new training set from support vectors (HOG + color histogram).
- **"1-4 PCA"**: `1-4_PCA.ipynb` - Develop a new training set from support vectors (PCA).

---

## Future Enhancements

- Further optimization of HOG and color histogram feature extraction.
- Integration with advanced feature extraction techniques (e.g., Deep Learning-based features).
- Deployment-ready Docker image for scalable use cases.

---

## Contributing

We welcome contributions from the community! Please follow the guidelines below:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The CIFAR-10 dataset, provided by [Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html)
- Open-source libraries and the developer community for their contributions.

---

For further details, feel free to contact [Abhinav Saurabh](mailto:your.email@example.com).

---
