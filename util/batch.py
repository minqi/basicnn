def create_batch_generator(data_x, data_y, batch_size):
	pairs = zip(data_x, data_y)
	assert(batch_size <= len(pairs))

	batch_start = 0
	while (True):
		next_start = batch_start + batch_size
		batch = pairs[batch_start:next_start]
		if next_start >= len(pairs):
			wrap = next_start - len(pairs)
			if wrap > 0:
				batch = np.concatenate((batch, batch[:wrap]), axis=0)
			next_start = wrap
		batch_start = next_start

		yield batch