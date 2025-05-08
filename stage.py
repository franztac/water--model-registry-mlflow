from mlflow.tracking import MlflowClient
import mlflow


client = MlflowClient()

model_name = "water_potability_rf"
model_version = 3
new_stage = "Production"


client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True,
)
