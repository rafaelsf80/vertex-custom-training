# Convert from CSV to TFRecord
import pandas as pd
import tensorflow as tf

from tensorflow.keras import utils
import pathlib
from pathlib import Path

def create_tf_example(text, label):

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8')])),
        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        #'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
    }))
    return tf_example


data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset = utils.get_file(
    'stack_overflow_16k.tar.gz',
    data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir='')
dataset_dir = pathlib.Path(dataset).parent

# Read train data as TFRecords
labels_dict = {'java':0, 'python':1, 'csharp':2, 'javascript':3}
with tf.io.TFRecordWriter("/Users/rafaelsanchez/git/vertex-customtraining-stackoverflow-EXTERNAL/dataset/stackoverflow-train.tfrecords") as writer:
    for train_directory in [dataset_dir/'train/java', dataset_dir/'train/python', dataset_dir/'train/csharp', dataset_dir/'train/javascript']:
        print("Generating: TRAIN ",train_directory.parts[-1], " ", len(list(train_directory.glob('*'))))
        for file in Path(train_directory).iterdir():
            if file.is_file():
                with open (file, "r") as myfile:
                    data = myfile.read().replace("\n", " ")
                myfile.close()
            # data: text; labels: int64, according to labels_dict
            example = create_tf_example(data, labels_dict[ str(train_directory.parts[-1]) ])
            writer.write(example.SerializeToString())
writer.close()

# Read train data as TFRecords
labels_dict = {'java':0, 'python':1, 'csharp':2, 'javascript':3}
with tf.io.TFRecordWriter("/Users/rafaelsanchez/git/vertex-customtraining-stackoverflow-EXTERNAL/dataset/stackoverflow-test.tfrecords") as writer:
    for test_directory in [dataset_dir/'test/java', dataset_dir/'test/python', dataset_dir/'test/csharp', dataset_dir/'test/javascript']:
        print("Generating: TEST ",test_directory.parts[-1])
        print("Generating: TEST ",len(list(test_directory.glob('*'))))
        for file in Path(test_directory).iterdir():
             if file.is_file():
                 with open (file, "r") as myfile:
                     data = myfile.read().replace("\n", " ")
                 myfile.close()
             example = create_tf_example(data, str(test_directory.parts[-1]))
             writer.write(example.SerializeToString())
writer.close()




# csv = pd.read_csv("/Users/rafaelsanchez/git/caip-edreams-workshop/data.csv").values
# with tf.io.TFRecordWriter("/Users/rafaelsanchez/git/caip-pipelines-tfx-stackoverflow-demo/dataset.tfrecords") as writer:
#   for row in csv:
#      features, label = row[:-1], row[-1]
#      print (features, label)
#      example = create_tf_example(features, label)
#      writer.write(example.SerializeToString())
# writer.close()