from typing import Text
import absl
import os
import tensorflow_model_analysis as tfma
import kfp
from tfx import v1 as tfx # !!


from tfx.components import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.proto import example_gen_pb2

PROJECT_ID         = 'ml-in-the-cloud-vertex'
REGION             = 'us-central1'
PIPELINE_NAME      = 'ml-in-the-cloud-course-stackoverflow-nlp-vertex'
PIPELINE_ROOT      = 'gs://ml-in-the-cloud-vertex-us/tfx-pipeline/stackoverflow-pipeline'
DATA_ROOT          = 'gs://ml-in-the-cloud-vertex-us/tfx-pipeline/stackoverflow-pipeline/data'
SERVING_MODEL_DIR  = 'gs://ml-in-the-cloud-vertex-us/tfx-pipeline/stackoverflow-pipeline/serving_model'
TFRECORDS_DIR_PATH = 'gs://ml-in-the-cloud-vertex-us/tfx-pipeline/tfrecord'
MODULE_FILE        = 'gs://ml-in-the-cloud-vertex-us/tfx-pipeline/stackoverflow_utils_vertex.py'

# We will use CPUs only for now.
USE_GPU = False

# Create a helper function to construct a TFX pipeline.
def create_vertex_pipeline(pipeline_name: str, input_dir: Text, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str, project_id: str,
                     region: str, use_gpu: bool) -> tfx.dsl.Pipeline:
   
    path_to_tfrecord_dir = input_dir

    # Output 2 splits: train:eval=3:1.
    output = example_gen_pb2.Output(
                split_config=example_gen_pb2.SplitConfig(splits=[
                    example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
                    example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
                ]))
    example_gen = ImportExampleGen(input_base=path_to_tfrecord_dir, output_config=output)
    
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    transform = Transform(
           examples=example_gen.outputs['examples'],
           schema=schema_gen.outputs['schema'],
           module_file=module_file)
  
    # NEW: Configuration for Vertex AI Training.
    # This dictionary will be passed as `CustomJobSpec`.
    vertex_job_spec = {
        'project': PROJECT_ID,
        'worker_pool_specs': [{
            'machine_spec': {
                'machine_type': 'n1-standard-4',
            },
            'replica_count': 1,
            'container_spec': {
                'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
            },
        }],
    }
    if use_gpu:
        # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
        # for available machine types.
        vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })

    # Trains a model using Vertex AI Training.
    # NEW: We need to specify a Trainer for GCP with related configs.
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=3000),
        eval_args=tfx.proto.EvalArgs(num_steps=3000),
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY:
                REGION,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                vertex_job_spec,
            'use_gpu':
                use_gpu,
        })


    # Set the TFMA config for Model Evaluation and Validation.
    eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='label')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          # Si el modelo no tiene una precisión mayor de 0.8
                          lower_bound={'value': 0.8}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-2})))
          ])
      ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        #baseline_model=model_resolver.outputs['model'],
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
           filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=SERVING_MODEL_DIR)))

    components=[
        example_gen, statistics_gen, schema_gen, example_validator, transform,
        trainer, evaluator, pusher
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=PIPELINE_ROOT,
        components=components
    )


# Compile and run the pipeline
print('TFX version: {}'.format(__import__('tfx.version').__version__))
print('KFP version: {}'.format(kfp.__version__))
print('TFMA version: {}'.format(tfma.__version__))
absl.logging.set_verbosity(absl.logging.INFO)


runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename='pipeline_dev_vertex.json')

_ = runner.run(
    create_vertex_pipeline(
        pipeline_name=PIPELINE_NAME,
        input_dir = TFRECORDS_DIR_PATH,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        module_file=MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
        project_id=PROJECT_ID,
        region=REGION,
        use_gpu=USE_GPU))

from kfp.v2.google import client

pipelines_client = client.AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

_ = pipelines_client.create_run_from_job_spec('pipeline_dev_vertex.json')