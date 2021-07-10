# Custom training job price prediction model with Hyperparameter tuning
# Training script located at 'script_custom_training_distributed.py'

from google.cloud import aiplatform

BUCKET = 'gs://ml-in-the-cloud-course'
PROJECT_ID = 'windy-site-254307'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET)

# Launch Training pipeline, a type of Vertex Training Job.
# A training pipeline integrates three steps into one job: Accessing a Managed Dataset (not used here), Training, and Model Upload. 
job = aiplatform.CustomTrainingJob(
    display_name="ml_in_the_cloud_4gpu_custom_training",
    script_path="script_custom_training_distributed.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-4:latest",
    requirements=['gcsfs==0.7.1'],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-4:latest",
)
model = job.run(
    model_display_name="ml_in_the_cloud_4gpu_custom_training",
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type= "NVIDIA_TESLA_K80",
    accelerator_count = 4
)
print(model)


# Deploy endpoint
endpoint = model.deploy(machine_type='n1-standard-4', 
    accelerator_type= "NVIDIA_TESLA_T4",
    accelerator_count = 1)
print(endpoint.resource_name)





