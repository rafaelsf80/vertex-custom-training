# Custom training pipeline, with script located at 'script_custom_training.py"'

from google.cloud import aiplatform

BUCKET = 'gs://ml-in-the-cloud-course'
PROJECT_ID = 'windy-site-254307'
SERVICE_ACCOUNT = 'prosegur-video-test@windy-site-254307.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE = 'projects/655797269815/locations/us-central1/tensorboards/3939734880274874368'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET)

# Launch Training pipeline, a type of Vertex Training Job.
# A Training pipeline integrates three steps into one job: Accessing a Managed Dataset (not used here), Training, and Model Upload. 
job = aiplatform.CustomTrainingJob(
    display_name="ml_in_the_cloud_custom_training_simple",
    script_path="script_custom_training.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-3:latest",
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest",
)
model = job.run(
    replica_count=1
)
print(model)

# Deploy endpoint
endpoint = model.deploy(machine_type='n1-standard-4')
print(endpoint.resource_name)





