from argparse import ArgumentParser
from os import environ, scandir

parser = ArgumentParser(
	description='Generates chat messages based on a message dataset',
)

parser.add_argument(
	'-l',
	'--length',
	type=int,
	dest='length',
	help='number of characters to generate',
	default=100,
)
parser.add_argument(
	'-m',
	'--model',
	required=True,
	choices=[f.name for f in scandir('trained') if f.is_dir()],
	dest='model',
	help='model to use',
)
parser.add_argument(
	'-t',
	'--text',
	required=True,
	type=str,
	dest='text',
	help='text to start the generation from',
)


def main():
	args = parser.parse_args()

	# disable TensorFlow logging
	environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# import main libraries after parsing arguments so it doesn't cause
	# unnecessary delay to incorrect input
	import tensorflow as tf
	from typing import cast

	from models import TextGenerator

	states: 'tf.RaggedTensor' | None = None
	next_char: 'tf.Tensor' = tf.constant([args.text])
	result = [next_char]

	generate = cast(
		'TextGenerator', tf.saved_model.load(f'trained/{args.model}/model')
	)

	for _ in range(args.length):
		next_char, states = generate.next_char(
			next_char, states=states
		)  # type: ignore
		result.append(next_char)

	print(tf.strings.join(result)[0].numpy().decode('utf-8'))


if __name__ == '__main__':
	main()
