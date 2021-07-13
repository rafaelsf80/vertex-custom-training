"""TXT based TFX example gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, Text

from absl import logging
import apache_beam as beam
import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _TxtToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[Text, Any],
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read Avro files and transform to TF examples.
  Note that each input split will be transformed by this function separately.
  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
      - input_base: input dir that contains Avro data.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.
  Returns:
    PCollection of TF examples.
  """
  input_base_uri = exec_properties[utils.INPUT_BASE_KEY]
  txt_pattern = os.path.join(input_base_uri, split_pattern)
  logging.info('Processing input TXT data %s to TFExample.', txt_pattern)

  return (pipeline
          | 'ReadFromTxt' >> beam.io.ReadFromText(txt_pattern)
          | 'ToTFExample' >> beam.Map(utils.dict_to_example))


class Executor(BaseExampleGenExecutor):
  """TFX example gen executor for processing avro format.
  Data type conversion:
    integer types will be converted to tf.train.Feature with tf.train.Int64List.
    float types will be converted to tf.train.Feature with tf.train.FloatList.
    string types will be converted to tf.train.Feature with tf.train.BytesList
      and utf-8 encoding.
    Note that,
      Single value will be converted to a list of that single value.
      Missing value will be converted to empty tf.train.Feature().
    For details, check the dict_to_example function in example_gen.utils.
  Example usage:
    from tfx.components.base import executor_spec
    from tfx.components.example_gen.component import
    FileBasedExampleGen
    from tfx.components.example_gen.custom_executors import
    avro_executor
    from tfx.utils.dsl_utils import external_input
    example_gen = FileBasedExampleGen(
        input=external_input(avro_dir_path),
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            avro_executor.Executor))
  """

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for avro to TF examples."""
    return _TxtToExample