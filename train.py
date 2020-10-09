# from Model import MyModel
import tensorflow as tf
from metrics import ROC,C_M
import datetime
from augmentation import augment
from plot import plot_roc
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)
import datetime
import matplotlib.image as mpimg


def training(train_ds, val_ds, test_ds, model, EPOCHS):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
            predictions, feature_map = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        pred_label = tf.math.argmax(predictions, axis=1)
        _ = train_AUC.update_state(labels, pred_label)
        # pred_axis0, pred_axis1 = tf.unstack(predictions, axis=1)
        # _ = train_ROC.update_state(labels, pred_axis0, pred_axis1)

    @tf.function
    def val_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        val_loss(v_loss)
        val_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        pred_label = tf.math.argmax(predictions, axis=1)
        _ = test_AUC.update_state(labels, pred_label)
        # _ = test_ROC.update_state(labels, predictions)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_AUC = tf.keras.metrics.AUC(
                                     num_thresholds=200, curve='ROC', summation_method='interpolation')
    train_ROC = ROC()
    num_class = 2
    # train_CM = C_M(num_class)

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_AUC = tf.keras.metrics.AUC(
                                    num_thresholds=200, curve='ROC', summation_method='interpolation')
    test_ROC = ROC()
    # test_CM = C_M(num_class)




    # EPOCHS = 10

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    ROC_log_dir = 'logs/gradient_tape/' + current_time + '/ROC'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    validation_summary_writer = tf.summary.create_file_writer(val_log_dir)
    ROC_summary_writer = tf.summary.create_file_writer(ROC_log_dir)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
       print("Restored from {}".format(manager.latest_checkpoint))
    else:
       print("Initializing from scratch.")

    for epoch in range(EPOCHS):
      train_loss.reset_states()
      train_accuracy.reset_states()
      train_AUC.reset_states()
  # train_CM.reset_states()
      train_ROC.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()
      test_AUC.reset_states()
      test_ROC.reset_states()
  # test_CM.reset_states()
      val_loss.reset_states()
      val_accuracy.reset_states()

      for sample in train_ds:
          size_shape = tf.random.uniform([], minval=200, maxval=256)
          train_img = sample[0]
          train_label = sample[2]
          train_img, train_label = augment(train_img, train_label, size_shape)
          train_step(train_img, train_label)
          predictions = model(train_img, training=True)

          label_pred = tf.math.argmax(predictions, axis=1)
          _ = train_ROC.update_state(train_label, predictions)
  #  _ = train_CM.update_state(train_label,label_pred)

      ckpt.step.assign_add(1)
           if int(ckpt.step) % 2 == 0:
              save_path = manager.save()
              print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
              print(manager.checkpoints)

    #  画ROC的图并保存
      fp, tp = train_ROC.result()
      plot_roc('ROC_train', fp, tp)  # create figure & 1 axis
      plt.savefig('ROC_train_img.png')  # save the figure to file
      plt.show()
      fig = mpimg.imread('ROC_train_img.png')
      fig = tf.expand_dims(fig, 0)

      with train_summary_writer.as_default():
           tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
           tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
           tf.summary.scalar('AUC_train', train_AUC.result(), step=epoch)
           tf.summary.image("ROC_train", fig, step=epoch)
  #  tf.summary.image('Confusion Matrix train', fig_CM, step=epoch)
      for sample in val_ds:
          val_img = sample[0]
          val_label = sample[2]
          val_step(val_img, val_label)
      with validation_summary_writer.as_default():
           tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
           tf.summary.scalar('val_accuracy', val_accuracy.result(), step=epoch)

      template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
      print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        val_loss.result(),
                        val_accuracy.result() * 100))

    for sample in test_ds:
        test_img = sample[0]
        test_label = sample[2]
        test_step(test_img, test_label)
        predictions = model(test_img, training=True)
        label_pred = tf.math.argmax(predictions, axis=1)
        _ = test_ROC.update_state(test_label, predictions)
  # _ = test_CM.update_state(test_label,label_pred)
#  画ROC的图并保存
    fp, tp = test_ROC.result()
    plot_roc('ROC_test', fp, tp)  # create figure & 1 axis
    plt.savefig('ROC_test_img.png')  # save the figure to file
    plt.show()
    fig = mpimg.imread('ROC_test_img.png')
    fig = tf.expand_dims(fig, 0)


    with test_summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss.result(), step=0)
        tf.summary.scalar('test_accuracy', test_accuracy.result(), step=0)
        tf.summary.scalar('AUC_test', train_AUC.result(), step=0)
        tf.summary.image("ROC_train", fig, step=0)
        # tf.summary.image('Confusion Matrix test', fig_CM, step=0)
        template = 'Test Loss: {}, Test Accuracy: {}'
        print(template.format(test_loss.result(),
                      test_accuracy.result() * 100))


