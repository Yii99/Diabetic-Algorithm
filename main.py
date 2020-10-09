from TFrecord import generator_TFrecord
from data_split import ds_func
from train import training
from Model import MyModel
data_dir = '/content/gdrive/My Drive/Dataset'
train_dir = data_dir + '/1. Original Images/a. Training Set/'
test_dir = data_dir + '/1. Original Images/b. Testing Set/'
gt_dir = data_dir + '/2. Groundtruths/'
train_label_dir = gt_dir+'a. IDRiD_Disease Grading_Training Labels.csv'
test_label_dir = gt_dir+'b. IDRiD_Disease Grading_Testing Labels.csv'

generator_TFrecord(train_dir, train_label_dir, 'train.tfrecords', oversampling=True)
generator_TFrecord(test_dir, test_label_dir, 'test.tfrecords', oversampling=False)

batch_size = 8
prefetch_size = 1000
train_ds, val_ds = ds_func('train.tfrecords', batch_size, prefetch_size, split=True)
test_ds = ds_func('test.tfrecords', batch_size, prefetch_size, split=False)

model = MyModel()
training(train_ds, val_ds, test_ds, model, EPOCHS=10)