# Vertex AI Training with Custom models

There are **three types** of resources to train custom models in Vertex AI:
1. Training pipelines
2. Custom jobs
3. Hyperparameter tuning jobs

## 1. Training pipelines

Resource to orchestrate a training pipeline that adds additional steps beyond training, such as:
* Loading a dataset 
* Uploading the resulting model to Vertex AI after the training job is successfully completed.

Training pipelines are implemented through the `aiplatform.CustomTrainingJob` class.

You can implement **distributed strategies** using this class.


## 2. Custom jobs

Resource to train an ML model in Vertex. You must specify:
* Training code
* Dependencies

Custom jobs are implemented through the `aiplatform.CustomJob` class.


## 3. Hyperparameter tuning jobs

Resource to implement hyperparameter tuning jobs. Based on the CustomJob class, but has additional settings to configure, such as the metric.

Hyperparameter tuning jobs are implemented through the `aiplatform.HyperparameterTuningJob` class.

Note that the HyperparameterTuningJob class **do not accept** a CustomTrainingJob instance as input.

You can combine distributed training with hyperparameter tuning.

# Vertex AI Training with TFX and Vertex Pipelines

A **Vertex pipeline** is not the same as a Training pipeline shown above. A Vertex pipeline can implement the full model workflow (from ingestion to deployment), while a training pipeline only focus on the "extended" training phase: download a dataset, training and upload the resulting model.

The `tfx-pipeline`directory shows how to run a Vertex pipeline in two training scenarios:
* Using `tfx.components.Trainer` class (file `pipeline.py`)
* Using `tfx.extensions.google_cloud_ai_platform.Trainer` class (file `pipeline_vertex.py`)

![Vertex pipelines result](tfx-pipeline/tfx-pipeline.png)

## Dataset

The dataset used for training is the Stackoverflow dataset at https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar. The Vertex pipelines code use the same dataset converted to **TFRecord format**.

## Resources
[1] [Custom training in Vertex](https://cloud.google.com/vertex-ai/docs/training/custom-training-methods)  
[2] [Codelab with MultiWorkerMirroredStrategy](https://codelabs.developers.google.com/vertex_multiworker_training)