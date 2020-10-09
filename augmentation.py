import tensorflow as tf

def augment(image, label, shape, batch_size):
  # image_aug = image[1]
  image_aug = tf.image.flip_left_right(image)
  # visualize(image, flipped)
  image_aug = tf.image.rot90(image_aug)
  # visualize(image, rotated)
  image_aug = tf.image.random_brightness(image_aug, max_delta=0.5) # Random brightness
  image_aug = tf.image.random_crop(image_aug, size=[batch_size, shape, shape, 3]) # Random crop back to 28x28
  image_aug = tf.image.resize(image_aug, [256,256])
  return image_aug,label