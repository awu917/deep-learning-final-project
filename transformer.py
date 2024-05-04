import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from dataset import *

def positional_encoding(length, depth):
    positions = tf.cast(tf.range(length)[:, tf.newaxis], dtype=tf.float32)
    depths = tf.cast(tf.range(depth)[tf.newaxis, :], dtype=tf.float32) / tf.cast(depth, tf.float32)
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1) 
    return pos_encoding

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x += self.pos_encoding
        return x

class SupernovaTransformer(tf.keras.Model):
    def __init__(self, sequence_len, output_dim, nb_classes, batch_size, num_heads=2, d_model=16, dff=16, dropout=0.1):
        super().__init__()

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # Adjust the dimension here
            tf.keras.layers.Dense(output_dim)
        ])
        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.nb_classes = nb_classes

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
        
        # Reshape the output to match the expected shape
        output = tf.reshape(soft, (-1,self.batch_size,self.nb_classes))
        print("Last output layer", output.shape)

        return output


    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """

        ## TODO: Implement similar to test below.

        ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##       range of indices spanning # of training entries, then tf.gather) 
        ##       to make training smoother over multiple epochs.

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0      
        num_batches = int(len(train_captions) / batch_size)
        total_loss = total_seen = total_correct = 0

        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_image_features = train_image_features[start:end, :]
            decoder_input = train_captions[start:end, :-1]
            decoder_labels = train_captions[start:end, 1:]
        
            with tf.GradientTape() as tape:
                probs = self(batch_image_features, decoder_input)
                mask = decoder_labels != padding_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(probs, decoder_labels, mask)

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

        avg_loss = float(total_loss / total_seen)
        avg_acc = float(total_correct / total_seen)
        avg_prp = np.exp(avg_loss)

        print(f"\r[Train {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print(f"\nTrain Epoch\t Loss: {avg_loss:.3f}\t Acc: {avg_acc:.3f}\t Perp: {avg_prp:.3f}")

        return avg_loss, avg_acc, avg_prp

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)


def accuracy_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Computes the batch accuracy

    :param prbs:  float tensor, word prediction probabilities [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
    :param labels:  integer tensor, word prediction labels [BATCH_SIZE x WINDOW_SIZE]
    :param mask:  tensor that acts as a padding mask [BATCH_SIZE x WINDOW_SIZE]
    :return: scalar tensor of accuracy of the batch between 0 and 1
    """
    correct_classes = tf.argmax(prbs, axis=-1) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Calculates the model cross-entropy loss after one forward pass
    Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: the loss of the model as a tensor
    """
    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss

def get_model(sequence_len, output_dim, nb_classes, batch_sz, optimizer_name, lr, epochs=1):
    optimizer = getattr(tf.keras.optimizers, optimizer_name)(learning_rate=lr)
    model = SupernovaTransformer(sequence_len, output_dim, nb_classes, batch_sz)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC()]
    )
    return SimpleNamespace(
        model=model,
        epochs=epochs,
        batch_size=batch_sz,
        optimizer=optimizer,
        lr=lr,
    )

def main():
    path = "data/unblind_hostz"
    test_fraction = 0.3
    classifier = sn1a_classifier
    
    (X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), (length_train, length_test, sequence_len, output_dim, nb_classes) = load_data(
        path=path, 
        test_fraction=test_fraction,
        classifier=classifier)
    
    args = get_model(sequence_len, output_dim, nb_classes, batch_sz=10, optimizer_name='Adam', lr=0.001, epochs=50)
    model = args.model
    
    Y_train_reshaped = np.expand_dims(Y_train, axis=1)
    Y_test_reshaped = np.expand_dims(Y_test, axis=1)

    compile_model(model, args.optimizer, 'categorical_crossentropy', [tf.keras.metrics.AUC()])
    
    # Train the model
    train_stats = train_model(model, X_train, Y_train, X_test, Y_test, args)

    # Test the model
    test_stats = test_model(model, X_test, Y_test, args)
    
    # Print or log training and testing statistics
    print("Training statistics:", train_stats)
    print("Testing statistics:", test_stats)


def compile_model(model, optimizer, loss, metrics):
    '''Compiles model by reference based on arguments'''
    model.optimizer = optimizer
    model.loss_function = loss 
    model.accuracy_function = metrics[0]

def train_model(model, train_captions, train_image_features, test_captions, test_image_features, args):
    '''Trains model and returns model statistics'''
    train_stats = []
    valid_stats = []
    
    for epoch in range(args.epochs):
        train_loss, train_acc, train_perp = model.train(train_captions, train_image_features, args.batch_size)
        train_stats.append((train_loss, train_acc, train_perp))

        valid_loss, valid_acc, valid_perp = model.test(test_captions, test_image_features, args.batch_size)
        valid_stats.append((valid_loss, valid_acc, valid_perp))

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.3f}, Acc={train_acc:.3f}, Perp={train_perp:.3f}; Valid Loss={valid_loss:.3f}, Acc={valid_acc:.3f}, Perp={valid_perp:.3f}")

    return train_stats, valid_stats

def test_model(model, test_captions, test_image_features, args):
    '''Tests model and returns model statistics'''
    test_loss, test_acc, test_perp = model.test(test_captions, test_image_features, args.batch_size)
    print(f"Test Loss={test_loss:.3f}, Acc={test_acc:.3f}, Perp={test_perp:.3f}")
    return test_loss, test_acc, test_perp
if __name__ == '__main__':
    main()
