import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from dataset import *
from matplotlib import pyplot as plt

class SupernovaTransformer(tf.keras.Model):

    def __init__(self, filter_size, nb_classes,hidden_sz=16, head_size=256, num_heads=4):
        super().__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads
        )
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=1)
        self.conv_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")
        self.dense1 = tf.keras.layers.Dense(units=hidden_sz)
        self.dense2 = tf.keras.layers.Dense(units=nb_classes)

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        attn_dropout = tf.nn.dropout(attn_output, 0.3)
        attn_layer_norm = self.attn_layer_norm(attn_dropout)
        res = attn_layer_norm + inputs

        conv1_output = self.conv1(res)
        conv1_dropout = tf.nn.dropout(conv1_output, 0.3)
        conv_layer_norm = self.conv_layer_norm(conv1_dropout)
        pooling = self.pooling(conv_layer_norm + res)
        logits1= self.dense1(pooling)
        relu_outputs = tf.nn.leaky_relu(logits1)
        dropout_outputs = tf.nn.dropout(relu_outputs, 0.3)
        logits2 = self.dense2(dropout_outputs)
        return tf.nn.softmax(logits2)

def get_model(filter_size, nb_classes, epochs=1, batch_sz=10):
    model = SupernovaTransformer(filter_size=filter_size, nb_classes=nb_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC()]
    )
    return SimpleNamespace(
        model=model,
        epochs=epochs,
        batch_size=batch_sz,
    )

def main():
    path = "data/unblind_hostz"
    test_fraction = 0.3
    classifier = sn1a_classifier
    
    (X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), (length_train, length_test, sequence_len, output_dim, nb_classes) = load_data(
        path=path, 
        test_fraction=test_fraction,
        classifier=classifier)

    args = get_model(filter_size=X_train.shape[-1], nb_classes=nb_classes, epochs=2, batch_sz=10)

    history = args.model.fit(
        X_train, Y_train,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X_test, Y_test)
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
