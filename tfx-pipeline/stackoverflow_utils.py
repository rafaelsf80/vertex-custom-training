from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

############################################################################################
# See MODULE_FILE constant at pipeline_dev.py to store this file
#############################################################################################

from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs

_FEATURE_KEY = 'text'
_LABEL_KEY = 'label'

_DROPOUT_RATE = 0.2
_EMBEDDING_UNITS = 64
_HIDDEN_UNITS = 64
_LEARNING_RATE = 1e-4
_L2_REGULARIZER=0.01
_VOCAB_SIZE = 8000
_MAX_LEN = 400
_TRAIN_BATCH_SIZE = 10
_EVAL_BATCH_SIZE = 5
_NUM_CLASSES = 4
_NUM_FILTERS=200
_FILTER_SIZE=4


#########################
## TRANSFORM COMPONENT ##
#########################
def _transformed_name(key, is_input=False):
  return key + ('_xf_input' if is_input else '_xf')

def _tokenize_text(text):
  print(text)
  text_sparse = tf.strings.split(tf.reshape(text, [-1])).to_sparse()
  # tft.apply_vocabulary doesn't reserve 0 for oov words. In order to comply
  # with convention and use mask_zero in keras.embedding layer, set oov value
  # to _VOCAB_SIZE and padding value to -1. Then add 1 to all the tokens.
  text_indices = tft.compute_and_apply_vocabulary(
      text_sparse, default_value=_VOCAB_SIZE, top_k=_VOCAB_SIZE)
  dense = tf.sparse.to_dense(text_indices, default_value=-1)
  # TFX transform expects the transform result to be FixedLenFeature.
  padding_config = [[0, 0], [0, _MAX_LEN]]
  dense = tf.pad(dense, padding_config, 'CONSTANT', -1)
  padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
  padded += 1
  return padded


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  return {
      _transformed_name(_LABEL_KEY):
          inputs[_LABEL_KEY],
      _transformed_name(_FEATURE_KEY, True):
          _tokenize_text(inputs[_FEATURE_KEY])
  }


#######################
## TRAINER COMPONENT ##
#######################
def _gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:

  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_transformed_name(_LABEL_KEY))

  return dataset

# Returns the output to be used in the serving signature.
def _get_serve_tf_examples_fn(model, tf_transform_output):

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files, tf_transform_output, batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files, tf_transform_output, batch_size=_EVAL_BATCH_SIZE)

  model = keras.Sequential([
      keras.layers.Embedding(_VOCAB_SIZE + 2,_EMBEDDING_UNITS,name=_transformed_name(_FEATURE_KEY)),
      keras.layers.Reshape((_MAX_LEN, _EMBEDDING_UNITS, 1)),
      keras.layers.Conv2D(_NUM_FILTERS,(_FILTER_SIZE,_EMBEDDING_UNITS),activation='relu',kernel_regularizer=keras.regularizers.l2(_L2_REGULARIZER)),
      keras.layers.Flatten(),
      keras.layers.Dropout(_DROPOUT_RATE),
      keras.layers.Dense(_HIDDEN_UNITS, activation='relu'),
      keras.layers.Dense(_NUM_CLASSES, activation ='softmax')
  ])

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=keras.optimizers.Adam(_LEARNING_RATE),
      metrics=['sparse_categorical_accuracy'])

  model.summary(print_fn=absl.logging.info)

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  # define the signature for serving
  signatures = {
      'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
              tf.TensorSpec(
                  shape=[None],
                  dtype=tf.string,
                  name='examples')),
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
