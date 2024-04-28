import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score
from dataset import *
from keras.layers import Input


class SupernovaRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, sequence_len, output_dim, model_type, hidden_size=16, rnn_size=16):

        super().__init__()
        self.sequence_len = sequence_len
        self.output_dim = output_dim
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
        logits1 = self.dense1(RNN_outputs)
        relu_outputs = tf.nn.leaky_relu(logits1)
        dropout_outputs = tf.nn.dropout(relu_outputs, 0.3)
        logits2 = self.dense2(dropout_outputs)
        probs = tf.nn.softmax(logits2)
        return probs


def get_model(sequence_len, output_dim, epochs = 1, batch_sz = 10, model_type = "vanilla"):

    model = SupernovaRNN(sequence_len, output_dim, model_type)

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
    
    args = get_model(sequence_len, output_dim, model_type = "gru", epochs = 50, batch_sz = 10)

    args.model.fit(
        X_train, Y_train,
        epochs= args.epochs, 
        batch_size= args.batch_size,
        validation_data=(X_test,Y_test)
    )

if __name__ == '__main__':
    main()
