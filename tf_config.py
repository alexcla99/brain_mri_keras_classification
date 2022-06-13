import tensorflow as tf

def tf_configure() -> None:
	"""Configure TensorFlow in order to enable GPU memory growth."""
	for gpu in tf.config.experimental.list_physical_devices("GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)