import tensorflow as tf

def ds_func(TF_filename, batch_size, prefetch_size, split=False):
    dataset = tf.data.TFRecordDataset(TF_filename)
    feature_description = {
        'feature0': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'feature1': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'feature2': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)
    count = 0
    for _ in parsed_dataset:
        count += 1

    def decode_func(sample):
        img = tf.io.decode_image(sample['feature0'], dtype=tf.float32)
        label = sample['feature1']
        img_name = sample['feature2']
        return (img, label, img_name)

    DATASET_SIZE = count
    train_size = int(0.85 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    parsed_dataset = parsed_dataset.map(decode_func)
    parsed_dataset = parsed_dataset.shuffle(count)
    parsed_dataset = parsed_dataset.prefetch(prefetch_size)
    train_dataset = parsed_dataset.take(train_size).batch(batch_size, drop_remainder=True)
    val_dataset = parsed_dataset.skip(train_size).batch(batch_size)

    if split:
        return train_dataset, val_dataset
    else:
        return parsed_dataset.batch(batch_size)