# 1. Importing Modules
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, SimpleRNN, Embedding, Dropout, Bidirectional, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.optimizers import Adam, RMSprop


# 2. Loading Data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
# print("Sentence before padding:\n {}".format(x_train[0]))

# 3. Formatting/Preparing Training Data
def pad_data_x_train(maxlen, x_train):
    x_train_padded = pad_sequences(x_train, maxlen=maxlen, truncating='post')

    # print("Sentence padded to length {}:\n {}".format(maxlen, x_train_padded[0]))

    # print(len(x_train_padded))

    return x_train_padded


# 4. Formatting/Preparing Test Data
def pad_data_x_test(maxlen, x_test):
    x_test_padded = pad_sequences(x_test, maxlen=maxlen, truncating='post')

    # print("Sentence padded to length {}:\n {}".format(maxlen, x_test_padded[0]))

    # print(len(x_test_padded))

    return x_test_padded


# 5. Model Declaration
def create_model(model_type='simple_RNN', embed_size=128, hidden_size=64, num_words=5000):
    model = Sequential()
    model.add(Embedding(num_words, embed_size))  # Embedding layer

    if model_type == 'simple_RNN':
        model.add(SimpleRNN(units=hidden_size, return_sequences=False))  # RNN layer
        model.add(Dropout(0.25))  # Dropout layer

    elif model_type == 'mono_LSTM':
        model.add(LSTM(units=hidden_size, return_sequences=False,
                       kernel_regularizer=l2(0.001)))  # Unidirectional LSTM with L2 regularization
        model.add(Dropout(0.4))  # Dropout layer

    elif model_type == 'bi_LSTM':
        model.add(Bidirectional(LSTM(units=hidden_size, return_sequences=False)))  # Bidirectional LSTM
        model.add(Dropout(0.4))  # Dropout layer

    elif model_type == 'multi_layer_LSTM':
        # Adding convolution and pooling layers
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))  # Convolution layer
        model.add(MaxPooling1D(pool_size=2))  # Pooling layer
        # model.add(Dropout(0.25))

        # Adding more convolution and pooling layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))  # Convolution layer
        model.add(MaxPooling1D(pool_size=2))  # Pooling layer
        # model.add(Dropout(0.25))

        # model.add(Bidirectional(LSTM(LSTM_size, return_sequences=True)))           # First bidirectional LSTM layer, returns the whole sequence
        # model.add(Bidirectional(LSTM(LSTM_size, return_sequences=True)))           # Second bidirectional LSTM layer, returns the whole sequence
        model.add(Bidirectional(LSTM(units=hidden_size,
                                     return_sequences=False)))  # Third bidirectional LSTM layer, returns the last time step's output
        # model.add(Dropout(0.25))                                                   # Dropout layer

    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.build((None, None))  # Model instantiation
    model.summary()  # Display model structure
    return model


# 6. Model Training
def compile_and_train(model, x_train, y_train, Batch_size=32, Epochs=30, model_type='simple_RNN',
                      Loss='binary_crossentropy'):
    # Configure callbacks
    Callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)]

    if model_type == 'simple_RNN':
        initial_learning_rate = 0.0001
        decay_rate = initial_learning_rate * 5 / Epochs
        Optimizer = Adam(learning_rate=initial_learning_rate, decay=decay_rate, clipvalue=0.5)
    elif model_type == 'mono_LSTM':
        Optimizer = 'adam'
    elif model_type == 'bi_LSTM':
        Optimizer = Adam(learning_rate=0.001)
    elif model_type == 'multi_layer_LSTM':

        Optimizer = RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08)

    # Compile model
    model.compile(loss=Loss, optimizer=Optimizer, metrics=['accuracy'])

    # Train model
    history = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_split=0.2,
                        callbacks=Callbacks)
    return history

# Plot output accuracy and loss curves
def plot_history(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Model Performance Evaluation
def evaluate_model(model, x_test, y_test):
    y_hat = model.predict(x_test)
    # Output value (predicted value) for each test sample

    # Transform predictions into labels
    i_pos = [i for i in range(len(y_hat)) if y_hat[i] > 0.5]
    i_neg = [i for i in range(len(y_hat)) if y_hat[i] <= 0.5]

    y_pred = np.zeros(len(y_hat))
    y_pred[i_pos] = 1
    y_pred[i_neg] = 0

    # Print continuous predicted values (probability values)
    print("y_hat :\n", y_hat)

    # Print converted predicted labels
    print("y_pred :\n", y_pred)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("Confusion Matrix:\n", cm)


# Compare accuracy and loss values across different models
def plot_training_histories(histories, metric='accuracy'):
    """
    Plot the training and validation metrics for multiple models in a single plot.

    Parameters:
    histories (dict): A dictionary where keys are model names and values are the history objects from model training.
    metric (str): The metric to plot, e.g., 'accuracy' or 'loss'.
    """
    plt.figure(figsize=(10, 6))
    for model_name, history in histories.items():
        plt.plot(history.history[metric], label=f'{model_name} train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'{model_name} val {metric}')

    plt.title(f'Model {metric.capitalize()} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

# Use different max_len values to create different training and test sets
max_len_100 = 100
max_len_500 = 500
max_len_1000 = 1000


x_train_len100 = pad_data_x_test(max_len_100,x_train)
x_test_len100 = pad_data_x_test(max_len_100,x_test)

x_train_len500 = pad_data_x_test(max_len_500,x_train)
x_test_len500 = pad_data_x_test(max_len_500,x_test)

x_train_len1000 = pad_data_x_test(max_len_1000,x_train)
x_test_len1000 = pad_data_x_test(max_len_1000,x_test)


# Run simple_RNN with max_len=100, batch size=16

model_simple_rnn_Len100_Batch16 = create_model('simple_RNN', embed_size=128, hidden_size=64)
history_simple_rnn_Len100_Batch16 = compile_and_train(model_simple_rnn_Len100_Batch16, x_train_len100, y_train, Batch_size=16, Epochs=20, model_type='simple_RNN')
evaluate_model(model_simple_rnn_Len100_Batch16, x_test_len100, y_test)
# plot_history(history_simple_rnn_Len100_Batch16)

# Run simple_RNN with max_len=100, batch size=32

model_simple_rnn_Len100_Batch32 = create_model('simple_RNN', embed_size=128, hidden_size=64)
history_simple_rnn_Len100_Batch32 = compile_and_train(model_simple_rnn_Len100_Batch32, x_train_len100, y_train, Batch_size=32, Epochs=20, model_type='simple_RNN')
evaluate_model(model_simple_rnn_Len100_Batch32, x_test_len100, y_test)
# plot_history(history_simple_rnn_Len100_Batch32)


# Run simple_RNN max_len=100, batch size=64

model_simple_rnn_Len100_Batch64 = create_model('simple_RNN', embed_size=128, hidden_size=64)
history_simple_rnn_Len100_Batch64 = compile_and_train(model_simple_rnn_Len100_Batch64, x_train_len100, y_train, Batch_size=64, Epochs=20, model_type='simple_RNN')
evaluate_model(model_simple_rnn_Len100_Batch64, x_test_len100, y_test)
# plot_history(history_simple_rnn_Len100_Batch64)

# Comparaison de simpleRNN utilisant différentes Batch size sous la même max_len
histories_1 = {
    'model_simple_rnn_Len100_Batch64': history_simple_rnn_Len100_Batch64,
    'model_simple_rnn_Len100_Batch32': history_simple_rnn_Len100_Batch32,
    'model_simple_rnn_Len100_Batch16': history_simple_rnn_Len100_Batch16,

}
plot_training_histories(histories_1, metric='accuracy')
plot_training_histories(histories_1, metric='loss')
