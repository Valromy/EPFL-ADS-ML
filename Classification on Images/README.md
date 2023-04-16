# Project Overview

## Task 1: Data Exploration and Feature Extraction
- Load and preprocess Swissroads dataset
- Plot images from each category
- Visualize category proportions in train, validation, and test sets
- (Optional) Plot color histograms for each category
- Extract high-level features using MobileNet v2
- (Optional) Visualize intensity of 1280 features for each category using heatmaps

## Task 2: PCA Analysis and Clustering
- Apply PCA analysis on training dataset
- Visualize transformed data on 2D-plot
- Apply k-means clustering (k=6)
- Transform test dataset using first two PCA components

## Task 3: Visual Search with k-NN
- Fit and tune k-NN classifier
- Provide classification report and visualize confusion matrix
- Plot 10 nearest neighbors for a correctly and misclassified image

## Task 4: Logistic Regression
- Train and evaluate logistic regression model
- Visualize model coefficients using heatmap
- Set "l2" regularization and tune regularization strength

## Task 5: Decision Trees and Random Forest
- Train decision tree with depth of 3
- Tune decision tree depth
- (Optional) Reduce dimensions with PCA
- Train and tune random forest model

## Task 6: Support Vector Machine (optional)
- Train SVM classifier with RBF and linear kernels
- Compute probabilities for each category for 10 images

## Task 7: Dense Network
- Implement 1-layer dense network
- Implement 2-layer dense network

## Task 8: Convolutional Neural Network
- Create ConvNet from scratch and train with pixel values

## Task 9: Final Comparison
- Collect test accuracy of all models
- Include a final visualization summarizing test accuracy
