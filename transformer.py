import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from dataset import *
from keras.layers import Input
from keras.utils import to_categorical

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
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x += self.pos_encoding
        return x

# class PositionalEncoding(tf.keras.layers.Layer):
#     def __init__(self, window_size, embed_size):
#         super().__init__()
#         self.embed_size = embed_size

#         ## Embed labels into an optimizable embedding space
#         self.embedding = tf.keras.layers.Embedding(input_dim=window_size, output_dim=embed_size)

#         ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
#         ## HINT: May want to use the function above...
#         self.pos_encoding = positional_encoding(window_size, embed_size)

    # def call(self, x):
    #     ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
    #     embedded = self.embedding(x)
    #     embedded *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        
    #     pos_encoding = positional_encoding(tf.shape(x)[1], self.embed_size)
        
    #     return embedded + pos_encoding



class SupernovaTransformer(tf.keras.Model):

    def __init__(self, sequence_len, output_dim, batch_size, num_heads=2, d_model=16, dff=16, dropout=0.1):
        super().__init__()

        self.ff_layer = tf.keras.Sequential([tf.keras.layers.Dense(2*output_dim, activation='relu'), 
                                             tf.keras.layers.Dense(output_dim)])
        self.self_atten         = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.output_dim = output_dim
        self.batch_size = batch_size

    def call(self, inputs):
    
        print("Input Shape:", inputs.shape)

        attention = self.self_atten(inputs, inputs, inputs)
        print("Attention Output Shape:", attention.shape)

        normalized_attn = self.layer_norm(inputs + attention)
        print("Normalized Attention Output Shape:", normalized_attn.shape)

        ff = self.ff_layer(normalized_attn)
        print("Feedforward Output Shape:", ff.shape)

        normalized_ff = self.layer_norm(ff)
        print("Normalized Feedforward Output Shape:", normalized_ff.shape)

        soft = tf.nn.softmax(normalized_ff)
        
        # Reshape the output to match the expected shape (None, 2, 13)
        # output = tf.reshape(soft, (-1, 2, 13))

        output = tf.reduce_mean(soft, axis=1)
        print("Last output layer", output.shape)

        
        return output

def get_model(sequence_len, output_dim, epochs=1, batch_sz=10):
    model = SupernovaTransformer(sequence_len, output_dim, batch_sz)
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

    Y_train_encoded = to_categorical(Y_train, num_classes=13)
    Y_train_encoded_reshaped = tf.reduce_mean(Y_train_encoded, axis=1)
    
    Y_test_encoded = to_categorical(Y_test, num_classes=13)
    Y_test_encoded_reshaped = tf.reduce_mean(Y_test_encoded, axis=1)

    batch_size = args.batch_size
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = Y_train_encoded_reshaped[i:i+batch_size]

            loss = args.model.train_on_batch(x_batch, y_batch)

            print(f"Batch {i//batch_size+1}/{len(X_train)//batch_size}: Loss = {loss}")

        # Reshape the ground truth labels to match the model output shape
        # Y_test_encoded_reshaped = Y_test_encoded.reshape(-1, 2, 13)
       
        
        val_loss = args.model.evaluate(X_test, Y_test_encoded_reshaped, batch_size=batch_size)
        print(f"Validation Loss: {val_loss}")

if __name__ == '__main__':
    main()
