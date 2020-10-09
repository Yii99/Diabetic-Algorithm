import tensorflow as tf
import pandas as pd




def generator_TFrecord(img_dir, csv_dir, file_name, oversampling=False):
    def image_func(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string)
        image = tf.cast(image_decoded, dtype=tf.float32) / 255.0
        image = tf.image.pad_to_bounding_box(image, 720, 0, 4288, 4288)
        image = tf.image.crop_to_bounding_box(image, 426, 298, 3435, 3435)
        image = tf.image.resize(image, (256, 256)) * 255.0
        # plt.figure()
        # plt.imshow(image)
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.io.encode_jpeg(image, quality=99)
        return image

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    df = pd.read_csv(csv_dir, usecols=[0, 1])
    type_dict = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
    df["Label"] = df['Retinopathy grade'].map(type_dict)
    df_class0 = df[df['Label'] == 0]
    df_class1 = df[df['Label'] == 1]
    df_nrdr = pd.concat([df_class0], sort=False)
    df_rdr = pd.concat([df_class1], sort=False)

    if oversampling:
        if len(df_nrdr) <= len(df_rdr):
            df_nrdr = df_nrdr.sample(len(df_rdr), replace=True)
        else:
            df_rdr = df_rdr.sample(len(df_nrdr), replace=True)

    df_over = pd.concat([df_nrdr, df_rdr], sort=False)
    df_over_shuffle = df_over.sample(frac=1)
    img_name = df_over_shuffle['Image name'].values
    img_label = df_over_shuffle['Label'].values

    def serialize_example(feature0, feature1, feature2):
        feature = {
            'feature0': _bytes_feature([feature0.numpy()]),
            'feature1': _bytes_feature([feature1.encode('utf_8')]),
            'feature2': _int64_feature(feature2),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(file_name) as writer:
        for num, i in enumerate(img_name):
            path = img_dir + i + '.jpg'
            img_train = image_func(path)
            label = img_label[num]
            example = serialize_example(img_train, i, label)
            if num % 20 == 0 or num == len(img_name):
                print(num)
            writer.write(example)
