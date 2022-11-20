import tensorflow as tf
from typing import cast


class TextModel(tf.keras.Model):
	def __init__(self, size: int, embedding_dim: int, rnn_units: int):
		super().__init__(self)

		self.embedding = tf.keras.layers.Embedding(size, embedding_dim)
		self.gru = tf.keras.layers.GRU(
			rnn_units, return_sequences=True, return_state=True
		)
		self.dense = tf.keras.layers.Dense(size)

	def call(self, inputs, states=None, return_state=None, training=False):
		x = inputs
		x = self.embedding(x, training=training)

		if states is None:
			states = self.gru.get_initial_state(x)

		x, states = cast(
			tuple['tf.Tensor', 'tf.RaggedTensor'],
			self.gru(x, initial_state=states, training=training)
		)
		x = self.dense(x, training=training)

		if return_state:
			return x, states
		else:
			return x


class TextGenerator(tf.keras.Model):
	def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
		super().__init__()
		self.temperature = temperature
		self.model = model
		self.chars_from_ids = chars_from_ids
		self.ids_from_chars = ids_from_chars

		skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
		sparse_mask = tf.SparseTensor(
			values=[-float('inf')] * len(skip_ids),
			indices=skip_ids,
			dense_shape=[len(ids_from_chars.get_vocabulary())]
		)
		self.prediction_mask = tf.sparse.to_dense(sparse_mask)

	@tf.function
	def next_char(
		self,
		inputs,
		states: 'tf.RaggedTensor' | None = None
	) -> tuple['tf.Tensor', 'tf.RaggedTensor' | None]:
		input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
		input_ids = self.ids_from_chars(input_chars).to_tensor()

		predicted_logits, states = self.model(
			inputs=input_ids, states=states, return_state=True
		)

		predicted_logits = predicted_logits[:-1:]
		predicted_logits = predicted_logits / self.temperature
		predicted_logits = predicted_logits + self.prediction_mask

		predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
		predicted_ids = tf.squeeze(predicted_ids, axis=-1)

		chars = cast('tf.Tensor', self.chars_from_ids(predicted_ids))

		return chars, states
