from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

############################################################################################
# See MODULE_FILE constant at pipeline_dev.py to store this file
#############################################################################################

_FEATURE_KEY = 'text'
_LABEL_KEY = 'label'

_DROPOUT_RATE = 0.2
_EMBEDDING_UNITS = 64
_EVAL_BATCH_SIZE = 5
_HIDDEN_UNITS = 64
_LEARNING_RATE = 1e-4
_L2_REGULARIZER=0.01
_LSTM_UNITS = 64
_VOCAB_SIZE = 8000
_MAX_LEN = 400
_TRAIN_BATCH_SIZE = 10
_NUM_CLASSES = 4
_NUM_FILTERS=200
_FILTER_SIZE=4

def transformed_name(key):
  return key + '_xf'


#######################
## TRANSFORM
########################

from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

from tfx.components.trainer.fn_args_utils import FnArgs

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
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  return {
      _transformed_name(_LABEL_KEY):
          inputs[_LABEL_KEY],
      _transformed_name(_FEATURE_KEY, True):
          _tokenize_text(inputs[_FEATURE_KEY])
  }


#######################
## TRAINER
########################
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


def _build_keras_model() -> keras.Model:

  model = keras.Sequential([
      keras.layers.Embedding(_VOCAB_SIZE + 2,_EMBEDDING_UNITS,name=_transformed_name(_FEATURE_KEY)),
      #keras.layers.Bidirectional(keras.layers.LSTM(_LSTM_UNITS, dropout=_DROPOUT_RATE)),
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
  return model


def _get_serve_tf_examples_fn(model, tf_transform_output):

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn




  
def general_run_fn(
    fn_args,
    label_key,
    training_args,
    train_batch_size=512,
    eval_batch_size=128,
    num_epochs=100
):
    """
    Train the model based on given args.
    This function is called by the Trainer component.

    :param fn_args: Holds args used to train the model as name/value pairs.
    :param label_key: column name representing the label.
    :param training_args: Dictionary holding all arguments expected by the
        _build_keras_model (according to the selected model).
    :param train_batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch in the training dataset.
    :param eval_batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch in the evaluation dataset.
    :param num_epochs: Number of epochs to run the training for.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, batch_size=_TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, batch_size=_EVAL_BATCH_SIZE)

    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model = _build_keras_model()


    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                                          tf.TensorSpec(
                                              shape=[None],
                                              dtype=tf.string,
                                              name='examples')),
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


def run_fn(fn_args):
    """Train the model based on given args.
    TFX Trainer will call this function

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    general_run_fn(
        fn_args,
        label_key=_LABEL_KEY,
        train_batch_size=_TRAIN_BATCH_SIZE,
        eval_batch_size=_EVAL_BATCH_SIZE,
        num_epochs=_NUM_EPOCHS,
        training_args=_TRAINING_ARGS
    )


def create_eval_config(
    metrics_specs,
    label_key=None,
    model_specs=None,
    slicing_specs=None
):
    """
    Parses a text protobuf into a tfma.EvalConfig object that is used in
    the Evaluator.

    :param metrics_specs: protobuf text with all metric specs and metric
        configs that should be calculated.
    :param label_key: column name representing the label. This is used to build
        the default model spec, in case it is not given.
    :param model_specs: protobuf text with the model spec. If model spec is
        given then the label_key argument is ignored.
    :param slicing_specs: protobuf text with the slicing spec.
    :return: tfma.EvalConfig
    """
    if not model_specs and not label_key:
        raise ValueError(
            "Label key must be defined if no custom model_specs is provided.")

    if not model_specs:
        model_specs = """
            model_specs {
                label_key: '""" + label_key + """'
            }
        """
    if not slicing_specs:
        slicing_specs = """
            slicing_specs {}
        """
    eval_config_specs = model_specs + slicing_specs + metrics_specs
    return text_format.Parse(eval_config_specs, tfma.EvalConfig())

"""Python source file including pipeline functions and necessary utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a
_run_fn function needs to be provided.  This file contains both.
"""

import tensorflow as tf
# TODO: if you use any additional functions from the common utils they should
#  be imported here


# TODO: Update constants used for training the model
_TRAIN_BATCH_SIZE = 128
_EVAL_BATCH_SIZE = 128
_NUM_EPOCHS = 1000

# TODO: Update configurations used for building the model (in function
#  _build_keras_model)
#  Configurations here should be passed to the model you choose.
_TRAINING_ARGS = {
    'dnn_hidden_units': [6, 3],
    'optimizer': tf.keras.optimizers.Adam,
    'optimizer_kwargs': {
        'learning_rate': 0.01
    }
}

# TODO: Update evaluation metrics configuration
_EVAL_METRIC_SPEC = """
    metrics_specs {
        metrics {
            class_name: "MeanSquaredError",
            threshold {
                value_threshold { upper_bound { value: 25 } }
                change_threshold {
                    absolute { value: 1 }
                    direction: LOWER_IS_BETTER
                }
            }
        }
        metrics {
            class_name: "Accuracy"
        }
    }
"""