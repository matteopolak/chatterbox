from argparse import ArgumentParser, ArgumentTypeError
from os import path, mkdir


def validate_file(file: str):
	if not path.exists(file):
		raise ArgumentTypeError(f'file {file} does not exist')

	return file


parser = ArgumentParser(
	description='Trains a model on a dataset of chat messages',
)

parser.add_argument(
	'-d',
	'--data',
	required=True,
	type=validate_file,
	dest='data',
	help='path to the csv file to train from (username, text)',
)
parser.add_argument(
	'-m',
	'--model',
	required=True,
	type=str,
	dest='model',
	help='name of the model',
)
parser.add_argument(
	'--seq-length',
	type=int,
	dest='seq_length',
	help='length of consecutive characters to train with',
	default=100,
)
parser.add_argument(
	'--batch-size',
	type=int,
	dest='batch_size',
	help='size of batch to use',
	default=256,
)
parser.add_argument(
	'--buffer-size',
	type=int,
	dest='buffer_size',
	help='size of buffer to use when shuffling data',
	default=10_000,
)
parser.add_argument(
	'--embedding-dim',
	type=int,
	dest='embedding_dim',
	help='dimension of tensor used to represent sequence',
	default=256,
)
parser.add_argument(
	'--rnn-units',
	type=int,
	dest='rnn_units',
	help='number of RNN units to use',
	default=1_024,
)
parser.add_argument(
	'--epochs',
	type=int,
	dest='epochs',
	help='number of epochs to train for',
	default=30,
)


def split_input_target(sequence: list[str]):
	return sequence[:-1], sequence[1:]


def main():
	args = parser.parse_args()

	# import main libraries after parsing arguments so it doesn't cause
	# unnecessary delay to incorrect input
	import pandas as pd
	import pickle
	import tensorflow as tf
	import os

	from models import TextModel, TextGenerator

	CHECKPOINT_PREFIX = os.path.join(
		'trained', args.model, 'checkpoints', 'ckpt_{epoch}'
	)
	CHECKPOINT_CALLBACK = tf.keras.callbacks.ModelCheckpoint(
		filepath=CHECKPOINT_PREFIX,
		save_weights_only=True,
		save_best_only=True
	)

	df = pd.read_csv(args.data, delimiter=',', names=['username', 'text'])
	text = (df.username.astype(str) + ': ' + df.text.astype(str) +
		'\n').str.split('').explode()
	text = text[text != '']

	try:
		vocab = sorted(
			pickle.load(open(f'trained/{args.model}/vocab.pickle', 'rb'))
		)
	except FileNotFoundError:
		chars = text.unique()
		vocab = sorted(set(chars))

		mkdir(f'trained/{args.model}')
		pickle.dump(vocab, open(f'trained/{args.model}/vocab.pickle', 'wb'))

	ids_from_chars = tf.keras.layers.StringLookup(
		vocabulary=list(vocab), mask_token=None
	)

	chars_from_ids = tf.keras.layers.StringLookup(
		vocabulary=ids_from_chars.get_vocabulary(),
		invert=True,
		mask_token=None
	)

	ids = ids_from_chars(text)
	ids_dataset = tf.data.Dataset.from_tensor_slices(ids)  # type: ignore
	sequences = ids_dataset.batch(args.seq_length + 1, drop_remainder=True)

	dataset = (
		sequences.map(split_input_target).shuffle(
		args.buffer_size
		).batch(args.batch_size,
		drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	)

	model = TextModel(
		size=len(ids_from_chars.get_vocabulary()),
		embedding_dim=args.embedding_dim,
		rnn_units=args.rnn_units
	)

	model.compile(
		optimizer='adam',
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
	)

	model.fit(dataset, epochs=args.epochs, callbacks=[CHECKPOINT_CALLBACK])

	generator = TextGenerator(model, chars_from_ids, ids_from_chars)
	next_char = tf.constant(['a'])
	states = None

	for _ in range(2):
		next_char, states = generator.next_char(
			next_char, states=states
		)  # type: ignore

	tf.saved_model.save(generator, f'trained/{args.model}/model')


if __name__ == '__main__':
	main()
