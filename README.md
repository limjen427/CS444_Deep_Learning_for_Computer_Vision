# CS444 Deep Learning for Computer Vision

## Overview
Welcome to the repository for my completed CS444 Deep Learning for Computer Vision class assignments. This repository showcases my understanding and application of various deep-learning concepts in the field of computer vision. Each assignment tackles different aspects of neural networks, from fundamental linear classifiers to advanced techniques like deep reinforcement learning and generative adversarial networks (GANs).

## Course Information
- **Programming Language**: Python (primarily PyTorch)
- **Class Goals**: This course provides a hands-on introduction to neural networks and deep learning, covering topics such as linear classifiers, multi-layer neural networks, backpropagation, stochastic gradient descent, convolutional neural networks(CNNs) for object detection and image labeling, generative models(GANs and diffusion models), sequence models like recurrent networks and transformers, applications of transformers in language and vision, and deep reinforcement learning.

## Assignments

### Assignment 1: Linear Classifiers
Implementing linear classifiers(logistic regression, perceptron, SVM, softmax) and applying them to two datasets: Rice dataset (binary classification) and Fashion-MNIST (multi-class image classification). Gain experience in hyperparameter tuning and proper data splits.

### Assignment 2: Multi-Layer Neural Networks
Implementing multi-layer neural networks and backpropagation for image reconstruction based on pixel coordinates. Train a four-layer network using SGD and Adam optimizers. Implement forward and backward passes to minimize mean square error (MSE) loss between original and reconstructed images. Explore different forms of input feature mapping as outlined in [this paper](https://bmild.github.io/fourfeat/) to improve image reconstruction.

### Assignment 3: Self-supervised and Transfer Learning, Object Detection
- Part 1: Self-supervised Learning on CIFAR10

  Utilize PyTorch to train a model on a self-supervised rotation prediction task on the CIFAR10 dataset. Fine-tune a subset of the model's weights and train the model in a fully supervised setting with different weight initializations. Implement ResNet18 architecture and perform image rotation prediction. Fine-tune the model's weights for CIFAR10 classification. Train the full network for CIFAR10 classification.
- Part 2: YOLO Object Detection on PASCAL VOC

  Implement a YOLO-like object detector on the PASCAL VOC 2007 dataset. Focus on implementing the loss function of YOLO and utilize the provided pre-trained network structure.

### Assignment 4: Cat Face Generation with GANs
Training a Generative Adversarial Network (GAN) on a cat dataset to generate cat face images. Gain experience in implementing GANs in PyTorch and augmenting natural images.

### Assignment 5: Deep Reinforcement Learning
Implementing Deep Q-Network (DQN) and Double DQN on Atari Breakout using OpenAI Gym. Understand deep reinforcement learning with pixel-level information and implement a recurrent state to encode history.

### Extra Assignment: Adversarial Attacks
Implementing adversarial attacks on pre-trained classifiers.
Basic Attack Methods: Implement Fast Gradient Sign (FGSM), Iterative Gradient Sign, and Least Likely Class Methods. 
Implement universal adversarial perturbations, defenses against adversarial attacks, robust training, robust architectures, preprocessing input images, and robustness to non-adversarial distribution change.

---

Thank you for visiting my repository. Feel free to explore each assignment for detailed implementation and results. If you have any questions or suggestions, please don't hesitate to reach out.
