# Vertex Pipelines demo (TFX) with Stackoverflow dataset

This demo is an alternative demo to the Taxi pipeline tutorial (structured data):
* Uses a different dataset:
    * Unstructured data (not tabular)
    * Text classification
* Uses TFX and Vertex Pipelines (uCAIP)
* TODO: Will offload to Vertex Training and Prediction managed services
* TODO: Will show other MLOps managed features

## Instructions

1. Create a Vertex notebook.
2. Download `pipeline_dev.py` and modify the following constants: PROJECT_ID, REGION, API_KEY, PIPELINE_NAME, PIPELINE_ROOT, TFRECORDS_DIR_PATH, MODULE_FILE 
3. Create an API key and modify `API_KEY` accordingly
4. Upload `stackoverflow_utils.py` to GCS and modify `MODULE_FILE` accordingly
3. Convert text files intro TFrecrod format with `txt2tfrecord.py`. Upload the resulting file into `TFRECORDS_DIR_PATH` on GCS
5. Configure `GOOGLE_APPLICATION_CREDENTIALS` environment variable and download the key, then run the pipeline with `python pipeline_dev.py`

## Demo script

1. The pipeline execution takes several minutes. Launch the pipeline before you make the preso.
2. Show the code: use Managed notebooks or your local dev environment, like VS Code.
3. Give an overview of the code. Show the different components: First component is ExampleGen to get the data; second component is StatiscGen about analyzes data and outputs statistics; ...
4. Show the different artifacts in the code
5. Go to the **Managed Pipelines UI** and check the execution of your pipeline. Show nodes, logs, ... on uCAIP.


## TFX intro

TFX makes it easier to implement MLOps by providing a toolkit that helps you orchestrate your ML process on various orchestrators, such as: Apache Airflow, Apache Beam, and Kubeflow Pipelines. TensorFlow Extended (TFX) is an end-to-end platform for deploying production ML pipelines, and provides the following:

* **TFX pipelines:** a pipeline is a portable implementation of an ML workflow. TFX pipelines is a toolkit for building ML pipelines. 

* **TFX libraries:** base functionality for many of the standard components.

* **TFX components:** components that you can use as a part of a pipeline.


## TFX Libraries

Libraries which provide the base functionality for many of the standard components. You can use the **TFX libraries** to add this functionality to your own custom components, or use them separately. Libraries include:

| ![Relationship between TFX libaries and pipeline components](https://www.tensorflow.org/tfx/guide/images/libraries_components.png) | 
|:--:| 
| *Figure: Relationship between TFX libaries and pipeline components (Source: tensorflow.org/tfx)* |

* **TensorFlow Data Validation (TFDV)** is a library for analyzing and validating machine learning data. It is designed to be highly scalable and to work well with TensorFlow and TFX.

* **TensorFlow Transform (TFT)** is a library for preprocessing data with TensorFlow.

* **TensorFlow Model Analysis (TFMA)** is a library for evaluating TensorFlow models. It is used along with TensorFlow to create an EvalSavedModel, which becomes the basis for its analysis. It allows users to evaluate their models on large amounts of data in a distributed manner, using the same metrics defined in their trainer. 

* **TensorFlow Metadata (TFMD)** provides standard representations for metadata that are useful when training machine learning models with TensorFlow. The metadata may be produced by hand or automatically during input data analysis, and may be consumed for data validation, exploration, and transformation. 

* **ML Metadata (MLMD)** is a library for recording and retrieving metadata associated with ML developer and data scientist workflows. Most often the metadata uses TFMD representations. MLMD manages persistence using SQL-Lite, MySQL, and other similar data stores.

* **TensorFlow Serving** is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs.

## Orchestration options

**Intermediate Representation (IR)** architecture compiles in two steps:
1. First step (front end compiler) compiles to the Protobuf based IR format. 
2. Second step compiles the IR to the native format supported by a given execution engine.

Managed Pipelines in uCAIP uses TFX IR:
* `ai_platform_pipelines.ai_platform_pipelines_dag_runner`
* `kubeflow.v2.kubeflow_v2_dag_runner`
Do not use TFX IR:
* `airflow.airflow_dag_runner`
* `beam.beam_dag_runner`
* `experimental.kubernetes.kubernetes_dag_runner`


## ML Metadata

[ML Metadata](https://github.com/google/ml-metadata) (MLMD) is an open-source library for recording and retrieving metadata associated with ML developer and data scientist workflows. MLMD is an integral part of TensorFlow Extended (TFX), but is designed so that it can be used independently.

MLMD tries to understand these questions:
* Which dataset did the model train on?
* What were the hyperparameters used to train the model?
* Which pipeline run created the model?
* Which training run led to this model?
* Which version of TensorFlow created this model?
* When was the failed model pushed?

Artifacts and components are grouped into an execution run.

MLMD registers the following types of metadata in a database called the Metadata Store.

1. Metadata about the artifacts generated through the components/steps of your ML pipelines. Example: which data generates that artifact
2. Metadata about the executions of these components/steps
3. Metadata about pipelines and associated lineage information. for example: resuts of two different model training

A very interesting functionality is **warm starting**, which allows not rerun the full pipeline. This is about **caching** some of the components that do not need to be retrained.

> Metadata in ML is the equivalent of logging in software development.


## TFX and Interactive notebooks

In 2019 Google announced [**TFX interactive context**](https://blog.tensorflow.org/2019/11/introducing-tfx-interactive-notebook.html), where you can build, debug, and run your TFX pipeline inside an interactive Google Colab or Jupyter notebook. Within this notebook environment, you can run **TFX component-by-component**, which makes it easier to iterate and experiment on your ML pipeline.


## TFX Pipelines and Components

### TFX Pipelines

A **pipeline** is composed of component instances and input parameters.
A TFX pipeline can run on various orchestrators, such as: Apache Airflow, Apache Beam, and Kubeflow Pipelines. 
Component instances produce artifacts as outputs and typically depend on artifacts produced by upstream component instances as inputs. 

A TFX pipeline includes several components, and each components has three elements: driver, executor and the publisher. 

* The **driver** queries the metadata store and supplies the resulting metadata to the executor.
* The **executor** is the one performing the processing. As ML engineer, you typically will write code here.
* The **publisher** accepts results of the executor and saves them in the metadata.

| ![TFX component](https://www.tensorflow.org/tfx/guide/images/component.png) | 
|:--:| 
| *Figure: TFX component (Source: tensorflow.org/tfx)* |

In a TFX pipeline, a unit of data, called an **artifact**, is passed between components. Normally a component has one input artifact ad one output artifact. Every artifact has an associated metadata that defines its type and properties.

A TFX pipeline is a **sequence of components** that implement an ML pipeline which is specifically designed for scalable, high-performance machine learning tasks. That includes modeling, training, serving inference, and managing deployments to online, native mobile, and JavaScript targets.

| ![Data between components](https://www.tensorflow.org/tfx/guide/images/prog_fin.png) | 
|:--:| 
| *Figure: Data between components (Source: tensorflow.org/tfx)* |

### TFX Components

* **ExampleGen** ingests input data and split input dataset
* **StatisticsGen** calculates the statistics for the dataset
* **SchemaGen** creates a data schema
* **ExampleValidator** looks for anomalies and missing values in the data
* **Transform** performs feature engineering in the dataset
* **Trainer** trains the model. Starting from TFX 0.14, training on AI Platform uses [custom containers](https://cloud.google.com/ml-engine/docs/containers-overview).You can specify a custom container in the `ai_platform_training_args` when creating the Runner at `kubeflow_dag_runner.KubeflowDagRunner`. If not specified, TFX will use a a public container image matching the installed version of TFX. Note that if you do specify a custom container, ensure the entrypoint calls into TFX's run_executor script (`tfx/scripts/run_executor.py`)
* **Resolver**: is a special component which handles special artifact resolution logics that will be used as inputs for downstream nodes. As inputs, you must pass an instance as well as a resolver strategy (for example: latest artifact, or latest blessed model) ...
* **Tuner** tunes the hyperparameters of the model
* **Evaluator** and **modelValidator** evaluates the performance of the model
* **Pusher** deploys the model on the serving infrastructure

> A component specification (or configurator) defines the component's input and output contract. These info is used by the driver and publisher. The specification specifies the component's input and output artifacts, and the parameters that are used for the component execution.
> Components have dependencies, component will collect data from metadata whenver available and put the result in the output
> Lineage and model provenance

### TFX Custom components

There are three types of custom components: 
* Python function-based components: easiest to build
* Container-based components
* Fully custom components


## Resources


