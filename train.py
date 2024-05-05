import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score
from dataset import *
from matplotlib import pyplot as plt


class SupernovaRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, model_type, hidden_size=16, rnn_size=16):

        super().__init__()

        self.rnn_size = rnn_size
        self.hidden_size = hidden_size

        if model_type.lower() == "lstm":
            self.RNNlayer = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=False)
        elif model_type.lower() == "gru":
            # idk if the return_sequences and return_state parameters are correct
            self.RNNlayer = tf.keras.layers.GRU(units = self.rnn_size, return_sequences=False, return_state=False)
        elif model_type.lower() == "vanilla":
            self.RNNlayer = tf.keras.layers.SimpleRNN(units = self.rnn_size, return_sequences=False, return_state=False)
        else:
            raise Exception("invalid model type")

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size)
        self.dense2 = tf.keras.layers.Dense(units=2)


    def call(self, inputs):
        RNN_outputs = self.RNNlayer(inputs)
        print("Output shape from recurrent layer:", RNN_outputs.shape)
        logits1 = self.dense1(RNN_outputs)
        relu_outputs = tf.nn.leaky_relu(logits1)
        dropout_outputs = tf.nn.dropout(relu_outputs, 0.3)
        logits2 = self.dense2(dropout_outputs)
        pred = tf.nn.softmax(logits2)
        return pred


def get_model(epochs = 1, batch_sz = 10, model_type = "vanilla"):

    model = SupernovaRNN(model_type)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC()]
    )

    return SimpleNamespace(
        model = model,
        epochs = epochs,
        batch_size = batch_sz,
    )


def main():

    path = "data/unblind_hostz"
    test_fraction = 0.3
    classifier = sn1a_classifier
    
    (X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), (length_train, length_test, sequence_len, output_dim, nb_classes) = load_data(
        path=path, 
        test_fraction=test_fraction,
        classifier=classifier)
    print("Input shape:", X_train.shape)
    print("Target label shape, y train:", Y_train.shape)
    print("Sequence length expected by model:", sequence_len)

    args = get_model(epochs = 50, batch_sz = 10, model_type = "gru")

    history = args.model.fit(
        X_train, Y_train,
        epochs = args.epochs, 
        batch_size = args.batch_size,
        validation_data=(X_test,Y_test)
    )

    #plotting acc and loss
    plt.plot(history.history['auc'], color='b')
    plt.plot(history.history['val_auc'], color='m')
    plt.title('model accuracy')
    plt.ylabel('AUC accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='m')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
