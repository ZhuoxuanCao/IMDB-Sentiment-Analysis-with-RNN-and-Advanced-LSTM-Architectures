

# IMDB Sentiment Analysis with RNN and Multi-Layer LSTM Architectures

This project explores sentiment analysis on the IMDB movie review dataset using various recurrent neural network (RNN) architectures, with a focus on multi-layer LSTM models. We investigate the performance of single-layer RNNs, unidirectional and bidirectional LSTMs, and combinations of convolutional layers (CNNs) with multi-layer LSTMs to improve accuracy and generalization.

## Project Overview

Sentiment analysis is a crucial application of natural language processing, widely used to understand public opinions in reviews, social media, and other forms of text data. In this project, we aim to analyze and improve the performance of different RNN and LSTM structures on the IMDB dataset. Through careful architecture selection and hyperparameter tuning, we achieve significant performance improvements by using multi-layer LSTMs and CNN-enhanced LSTM models.

## Network Architectures

1. **Simple RNN**: A basic RNN model, useful for testing sequential dependencies but limited in its ability to capture long-term dependencies.
2. **Unidirectional LSTM**: A single-layer LSTM that processes sequences from past to future, suitable for capturing long-term dependencies but lacking bidirectional context.
3. **Bidirectional LSTM (Bi-LSTM)**: An LSTM with two directional flows—one from past to future and one from future to past—enabling the model to capture both preceding and following contextual information, improving accuracy on sentiment analysis tasks.
4. **Multi-Layer LSTM**: A stacked LSTM architecture with multiple layers, enhancing the model’s ability to capture complex, long-term dependencies in the text data.
5. **CNN + Multi-Layer LSTM**: This architecture combines two CNN layers with a multi-layer LSTM, enabling local feature extraction followed by sequential dependency capture. This combination not only improves accuracy but also reduces computational costs and training time.

## Experimental Setup

- **Dataset**: IMDB movie reviews with binary sentiment labels (positive or negative).
- **Hyperparameters**:
  - `num_words = 5000`: Vocabulary size, selecting the 5000 most frequent words.
  - `embed_size = 128`: Dimension size for word embeddings.
  - `Dropout = 0.4`: Dropout rate to prevent overfitting.
  - `Batch size`: Experimented with 16, 32, and 64 to balance training stability and model generalization.
  - `maxlen`: Sequence length, tested at 100, 500, and 1000 for optimal contextual capture.
  - **Optimizer**: Adam with a learning rate of 0.0001, utilizing gradient clipping and decay to stabilize training.
  - **Epochs**: 20, ensuring the model fully learns without overfitting.

## Results and Performance Comparison

### Model Performance Summary

1. **Simple RNN**: High training accuracy but significant fluctuations on the validation set, indicating severe overfitting.
2. **Unidirectional LSTM**: Moderate performance, but it lacks the bidirectional context needed for more comprehensive sentiment analysis.
3. **Bidirectional LSTM (Bi-LSTM)**: Achieves higher accuracy and better generalization by capturing both past and future context within sequences.
4. **Multi-Layer LSTM**: Provides the best test set results due to its increased capacity for long-term dependency capture, though at the cost of increased training complexity.
5. **CNN + Multi-Layer LSTM**: Outperforms other models in both accuracy (0.867) and training efficiency, reducing epoch time from 600 minutes to 36 minutes by leveraging CNN layers for local feature extraction before passing data to LSTM layers.

### Model Loss and Accuracy Comparisons

The following summarizes how different architectures performed across training and validation sets:
- **Shorter Sequence Lengths** (e.g., 100) increased training speed but led to overfitting, limiting generalization.
- **Longer Sequence Lengths** (e.g., 500 or 1000) enhanced contextual information capture, reducing validation loss and improving accuracy.
- **Larger Batch Sizes** reduced overfitting but required balanced selection to maintain training stability and performance.

## Conclusion

For tasks requiring extensive contextual information capture and high accuracy, the CNN + Multi-Layer LSTM combination is recommended. This architecture enables effective local feature extraction and sequential dependency learning, achieving a strong balance between accuracy and computational efficiency. Given the size and nature of the IMDB dataset, adding CNN layers improves the model's generalization ability and training speed, significantly outperforming standalone LSTM models.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 
- Necessary Python libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib` for plotting


### Usage

1. **Data Preparation**: Download and preprocess the IMDB dataset.
2. **Model Training**: Run the training script with different model configurations to evaluate performance.
3. **Evaluation**: Compare model performance metrics such as accuracy and loss on the validation set.


## References
[1] Qianzi Shen, Zijian Wang, Yaoru Sun. Sentiment Analysis of Movie Reviews Based on CNN-BLSTM.2nd International Conference on Intelligence Science (ICIS), Oct 2017, Shanghai, China. pp.164-171,ff10.1007/978-3-319-68121-4_17ff. ffhal-01820937f
([https://link-to-paper](https://link.springer.com/chapter/10.1007/978-3-319-68121-4_17))

