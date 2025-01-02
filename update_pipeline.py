from kfp import Client
from kfp.dsl import pipeline
from kfp.components import load_component_from_file

# Connect to the Kubeflow Pipelines API
KUBEFLOW_HOST = 'http://<kubeflow-host>:8080'
client = Client(host=KUBEFLOW_HOST)

# Define a pipeline
@pipeline(name="My Updated Pipeline", description="Pipeline updated by Jenkins")
def my_pipeline():
    pass

# Compile and upload the pipeline
pipeline_file = 'updated_pipeline.yaml'
client.create_pipeline_from_pipeline_func(my_pipeline, pipeline_filename=pipeline_file)
client.upload_pipeline(pipeline_file, pipeline_name='Updated Pipeline')
