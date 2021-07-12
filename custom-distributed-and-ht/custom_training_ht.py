# Custom training job price prediction model with Hyperparameter tuning
# Training script located at 'script_custom_training_ht.py'

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

BUCKET = 'gs://ml-in-the-cloud-course'
PROJECT_ID = 'windy-site-254307'
SERVICE_ACCOUNT = 'prosegur-video-test@windy-site-254307.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE = 'projects/655797269815/locations/us-central1/tensorboards/3939734880274874368'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET)

# Launch Training Job
job = aiplatform.CustomJob.from_local_script(display_name='ml_in_the_cloud_custom_ht_training_tb', 
        script_path='script_custom_training_ht.py',
        container_uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-4:latest',
        requirements=['gcsfs==0.7.1', 'cloudml-hypertune'],
        #model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-4:latest',
        machine_type="n1-standard-4",
        accelerator_type= "NVIDIA_TESLA_K80",
        accelerator_count = 4)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name='ml_in_the_cloud_custom_ht_training_tb',
    custom_job=job,
    metric_spec={'accuracy': 'maximize'},
    parameter_spec={
        'lr': hpt.DoubleParameterSpec(min=0.01, max=0.1, scale='log'),
        'units': hpt.IntegerParameterSpec(min=8, max=32, scale='linear'), # not used
        'activation': hpt.CategoricalParameterSpec(values=['relu', 'tanh']),  
        'batch_size': hpt.DiscreteParameterSpec(values=[32, 64, 128], scale='linear')
    },
    max_trial_count=16,
    parallel_trial_count=4,    
    )

hp_job.run( 
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE)