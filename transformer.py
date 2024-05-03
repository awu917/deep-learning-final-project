import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from dataset import *
from keras.layers import Input

def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    ## TODO: Can remove signature
    ## Generate a range of positions and depths 
        positions = tf.cast(tf.range(length)[:, tf.newaxis], dtype=tf.float32)    # (seq, 1)
        depths = tf.cast(tf.range(depth)[tf.newaxis, :], dtype=tf.float32) / tf.cast(depth, tf.float32)  # (1, depth)
        
        ## Compute range of radians to take the sine and cosine of.
        angle_rates = 1 / (10000 ** depths)               # (1, depth)
        angle_rads = positions * angle_rates             # (pos, depth)
        
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1) 
        ## This serves as offset for the Positional Encoding
        return pos_encoding


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, window_size, embed_size):
        super().__init__()
        self.embed_size = embed_size

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(input_dim=window_size, output_dim=embed_size)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        embedded = self.embedding(x)
        embedded *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        
        pos_encoding = positional_encoding(tf.shape(x)[1], self.embed_size)
        
        return embedded + pos_encoding



class SupernovaTransformer(tf.keras.Model):

    def __init__(self, sequence_len, output_dim, num_heads=2, d_model=16, dff=16, dropout=0.1):
        super().__init__()
        self.sequence_len = sequence_len
        self.output_dim = output_dim

        self.embedding = PositionalEncoding(sequence_len, d_model)
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        embedded = self.embedding(inputs)
        attention_output = self.multi_head_attention(embedded, embedded)
        pos_encoding = self.embedding.pos_encoding[:, :tf.shape(embedded)[1], :]
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + pos_encoding)
        ffn_output = self.dense2(self.dense1(attention_output))
        return tf.nn.softmax(ffn_output)

def get_model(sequence_len, output_dim, epochs=1, batch_sz=10):
    model = SupernovaTransformer(sequence_len, output_dim)
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
    
    args = get_model(sequence_len, output_dim, epochs=50, batch_sz=10)

    args.model.fit(
        X_train, Y_train,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X_test, Y_test)
    )

if __name__ == '__main__':
    main()
