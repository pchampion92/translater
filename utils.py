from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import tensorflow_text
from constants import MODEL_NAME, MAX_TOKENS, BUFFER_SIZE, BATCH_SIZE


def display_sample(dataset, example_size=10):
    for pt_examples, en_examples in dataset.batch(example_size).take(1):
        for pt_sentence in pt_examples.numpy():
            print(pt_sentence.decode('utf-8'))

        for en_sentence in en_examples.numpy():
            print(en_sentence.decode('utf-8'))


def load_tokenizer(model_name, cache_dir=Path('.')):
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir=cache_dir, cache_subdir='', extract=True
    )
    tokenizer = tf.saved_model.load(cache_dir / model_name)
    return tokenizer


def plot_distribution_sentence_length(dataset, model_name=MODEL_NAME):
    tokenizers = load_tokenizer(model_name)
    lengths = {'pt': [], 'en': []}
    for pt_example, en_example in dataset.batch(1024):
        pt_tokens = tokenizers.pt.tokenize(pt_example)
        en_tokens = tokenizers.en.tokenize(en_example)
        lengths['pt'].append(pt_tokens.row_lengths())
        lengths['en'].append(en_tokens.row_lenghts())
        print('.', end='', flush=False)
    lengths['pt'] = np.concatenate(lengths['pt'])
    lengths['en'] = np.concatenate(lengths['en'])
    lengths = pd.DataFrame.from_dict(lengths)
    fig = px.histogram(lengths, barmode='overlay')
    fig.show()


def filter_max_tokens(pt_tokens, en_tokens):
    num_tokens = tf.maximum(tf.shape(pt_tokens)[1], tf.shape(en_tokens)[1])
    return num_tokens < MAX_TOKENS


def tokenize_pairs_builder(tokenizer):
    def tokenize_pairs(pt, en):
        pt_tokens = tokenizer.pt.tokenize(pt)
        en_tokens = tokenizer.en.tokenize(en)
        # Convert ragged tensors to dense tensors (padded with zeros)
        pt_tokens = pt_tokens.to_tensor()
        en_tokens = en_tokens.to_tensor()
        return pt_tokens, en_tokens

    return tokenize_pairs


def make_batches(dataset, model_name=MODEL_NAME):
    tokenizer = load_tokenizer(model_name)
    tokenize_pairs = tokenize_pairs_builder(tokenizer)
    return (
        dataset
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs)
            .filter(filter_max_tokens)
            .prefetch(tf.data.AUTOTUNE)
    )


def get_angles(pos, i, embedding_dim):
    return pos / (10000 ** (2 * (i // 2) / float(embedding_dim)))


def positional_encoding(position, embedding_dim):
    pos_encoding = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(embedding_dim)[np.newaxis, :],
        embedding_dim
    )
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    # Mask all padded token
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    # Mask future elements in a sequence
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # Embedding dimension
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
