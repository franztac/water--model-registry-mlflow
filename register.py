from mlflow.tracking import MlflowClient
import mlflow


client = MlflowClient()

run_id = "ed17759d2f984111b55c145c3c25108e"
model_path = "mlflow-artifacts:/433294006654948739/ed17759d2f984111b55c145c3c25108e/artifacts/Best Model"
model_name = "water_potability_rf"

model_uri = f"runs:/{run_id}/{model_path}"

reg = mlflow.register_model(model_uri, model_name)
