# **Image Captioning Using Deep Learning**

This project implements an **Image Captioning** system using deep learning techniques. The model extracts features from images using a **pretrained CNN** and generates captions using an **LSTM-based decoder** and attention mechanism.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset and Preprocessing](#dataset-and-preprocessing)
3. [Model Training](#model-training)
4. [Results](#results)
5. [Setup and Usage](#setup-and-usage)
6. [Acknowledgements](#acknowledgements)

---

## **Overview**

This project is divided into two main stages:

1. **Feature Extraction**: Using a **pretrained CNN (Resnet50)** to extract visual features from images.
2. **Caption Generation**: Training an **LSTM-based language model** to generate textual descriptions for images.
3. **Text Embedding**:  
   - Utilizes **GloVe (Global Vectors for Word Representation)** for textual feature encoding.  
4. **Attention-Based Caption Generation**:  
   - Employs a **softmax attention mechanism** to enhance focus on relevant image regions.

### **Key Features**
- **ResNet Pretrained Model** for high-level image feature extraction.  
- **GloVe Embeddings** for efficient word representation.  
- **Attention Mechanism** to improve caption quality.  

This project is trained using cloud-based resources to handle large datasets and computational constraints.

---

## **Dataset and Preprocessing**

### **Dataset**
The model is trained on the **Flickr8K dataset**, which contains:
- **8,000 images**
- **40,000 captions** (5 per image)

### **Preprocessing Steps**
- **Image Features Extraction**: Uses **ResNet** to generate **feature vectors** for each image.  
- **Text Embeddings**:  
  - Reads **GloVe 50D embeddings** to represent words as vectors.  
  - Builds a vocabulary mapping **words to numerical embeddings**.  
- **Tokenization and Padding**:  
  - Converts captions to sequences.  
  - Pads sequences to ensure uniform length.
---

## **Model Training**

The model was trained in **Google Colab** using the following setup:

### **Training Setup**
- **Architecture**:
  - **Feature Extractor**: **ResNet50** (pretrained on ImageNet) with the classification head removed.
  - **Caption Generator**: **LSTM-based** sequence model for text generation.
  - **Embedding Layer**: Uses **GloVe embeddings** (pretrained word vectors) to represent words in the captions.
  - **Attention Mechanism**: Focuses on relevant image features while generating each word in the caption.
  
- **Loss Function**:  
  - **Categorical Cross-Entropy** (for multi-class word prediction).  

- **Optimizer**:  
  - **Adam Optimizer** with adaptive learning rate.  

- **Batch Size**: `32`  
- **Epochs**: `50`  




---



---

## **Setup and Usage**

Follow these steps to set up and use the project:

### **Installation**
```bash
git clone https://github.com/your_username/image-captioning.git
cd image-captioning
pip install -r requirements.txt
