import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from dataset import *
from keras.layers import Input

class SupernovaTransformer(tf.keras.Model):

    def __init__(self, last_filter_size, nb_classes,head_size=256, num_heads=4, ff_dim=4,dropout=0.25):
        super().__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.conv1 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
        self.conv_dropout = tf.keras.layers.Dropout(dropout)
        self.conv2 = tf.keras.layers.Conv1D(filters=last_filter_size, kernel_size=1)
        self.conv_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")
        self.dense = tf.keras.layers.Dense(units=nb_classes)


    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        attn_dropout = self.attn_dropout(attn_output)
        attn_layer_norm = self.attn_layer_norm(attn_dropout)
        res = attn_layer_norm + inputs

        conv1_output = self.conv1(res)
        conv1_dropout = self.conv_dropout(conv1_output)
        conv2_output = self.conv2(conv1_dropout)
        conv_layer_norm = self.conv_layer_norm(conv2_output)
        pooling = self.pooling(conv_layer_norm + res)
        dense= self.dense(pooling) 
        return tf.nn.softmax(dense)

def get_model(last_filter_size, nb_classes, epochs=1, batch_sz=10):
    model = SupernovaTransformer(last_filter_size=last_filter_size, nb_classes=nb_classes)
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

    args = get_model(last_filter_size=X_train.shape[-1], nb_classes=nb_classes, epochs=10, batch_sz=10)

    args.model.fit(
        X_train, Y_train,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X_test, Y_test)
    )

if __name__ == '__main__':
    main()
