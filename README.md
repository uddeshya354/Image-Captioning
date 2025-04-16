# **🖼️ Image Captioning Using Deep Learning**

This project implements an **Image Captioning** system using deep learning techniques. The model extracts features from images using a **pretrained CNN** and generates captions using an **LSTM-based decoder** with an **attention mechanism**.

---

## 📑 **Table of Contents**
1. [Overview](#overview)
2. [Dataset and Preprocessing](#dataset-and-preprocessing)
3. [Model Training](#model-training)
4. [Results](#results)
5. [Setup and Usage](#setup-and-usage)
6. [Acknowledgements](#acknowledgements)

---

## 📖 **Overview**

This project consists of four major stages:

1. **Feature Extraction**  
   - Uses a **pretrained ResNet50 CNN** to extract high-level image features.

2. **Text Embedding**  
   - Utilizes **GloVe (Global Vectors for Word Representation)** embeddings to encode words into dense vectors.

3. **Caption Generation**  
   - An **LSTM-based language model** is trained to generate captions for images.

4. **Attention-Based Caption Generation**  
   - Incorporates a **softmax attention mechanism** that enables the model to focus on relevant regions of the image when predicting each word.

### ✨ **Key Features**
- **ResNet50** pretrained on ImageNet for high-quality image feature extraction.
- **GloVe 50D word embeddings** for rich word representations.
- **Attention Mechanism** to enhance caption generation by dynamically focusing on different parts of the image.
- Trained using **Google Colab** to leverage cloud computing resources.

---

## 🗂️ **Dataset and Preprocessing**

### 📚 **Dataset**
The model is trained on the **Flickr8K dataset**, which contains:
- **8,000 images**
- **40,000 captions** (5 per image)

### ⚙️ **Preprocessing Steps**
- **Image Features Extraction**
  - Pass images through **ResNet50** to extract **feature vectors**.
  
- **Text Embeddings**
  - Load **GloVe 50D embeddings**.
  - Create a vocabulary mapping **words to vector indices**.
  
- **Tokenization and Padding**
  - Convert captions to sequences of integers using a tokenizer.
  - Apply padding to sequences to ensure consistent length across batches.

---

## 🏋️ **Model Training**

The model is trained in **Google Colab** using the following configuration:

### 📐 **Architecture**
- **Feature Extractor**
  - **ResNet50** pretrained on ImageNet, with the classification head removed.
  
- **Embedding Layer**
  - Uses **pretrained GloVe 50D embeddings**.

- **Caption Generator**
  - **LSTM-based** sequence model to predict the next word.

- **Attention Mechanism**
  - Focuses on image feature vectors during word generation.

### 🔍 **Training Parameters**
- **Loss Function**
  - **Categorical Cross-Entropy** (for multi-class word prediction).
  
- **Optimizer**
  - **Adam Optimizer** with adaptive learning rate.

- **Batch Size**: `32`
- **Epochs**: `50`

---

## 🎨 **Results**

- **Qualitative Results**
  - The model generates descriptive captions for unseen images.
  - Attention maps highlight image regions the model focuses on during word prediction.



## 🛠️ **Setup and Usage**

Follow these steps to set up and use the project:

### 📦 Installation
```bash
git clone https://github.com/your_username/image-captioning.git
cd image-captioning
pip install -r requirements.txt

