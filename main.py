import logging
import time

import tensorflow as tf
import tensorflow_datasets as tfds

from constants import MODEL_NAME, BATCH_SIZE, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, FORWARD_DIM, DROPOUT_RATE, EPOCHS
from schedule import CustomSchedule
from transformer import Transformer
from translator import Translator
from utils import display_sample, load_tokenizer, make_batches, positional_encoding, loss_function, accuracy_function

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# Read a sample of three sentences.
display_sample(train_examples)

# Tokenization (words or subwords tokenization, tracked with an ID, to be converted in embeddings)
tokenizers = load_tokenizer(MODEL_NAME)

### Sample tokens
print([item for item in dir(tokenizers.en) if not item.startswith('_')])

# Compute the distribution of length of sentences in the train set
if False:
    plot_distribution_sentence_length(train_examples)

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

pos_encoding = positional_encoding(BATCH_SIZE, EMBEDDING_DIM)

learning_rate = CustomSchedule(EMBEDDING_DIM)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    dff=FORWARD_DIM,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    rate=DROPOUT_RATE)

checkpoint_path = './checkpoints/train'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'
translator = Translator(tokenizers, transformer)
translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)
