import mlflow
from utils import load_config, setup_logging


logger = setup_logging()

def promote_best_model_to_production():
    config = load_config()
    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    model_name = config["model"]["name"]

    models = mlflow.search_registered_models()

    # Get the current model in Production
    current_prod_model = client.get_latest_versions(model_name, stages=["Production"])

    # If no model is in production, directly promote the best new model
    if len(current_prod_model) == 0:
        logger.info("No model in production. Searching for the best new model to promote.")
        best_run = None
        best_accuracy = 0.0
        experiments = client.search_experiments()

        for exp in experiments:
            runs = client.search_runs(exp.experiment_id, filter_string="status = 'FINISHED'")  # Only get finished runs
            for run in runs:
                acc = float(run.data.metrics.get("accuracy", 0))
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_run = run

        if best_run:
            model_uri = f"runs:/{best_run.info.run_id}/{model_name}"
            model_name = "HeartDiseaseModel"

            # Register the model
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)

            # Transition to Production
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Production"
            )

            logger.info(f"Promoted model version {mv.version} with accuracy {best_accuracy}")

    else:
        # Get the latest model's accuracy
        latest_run = mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]
        logger.info("latest_runlatest_run"), latest_run
        latest_accuracy = latest_run["metrics.accuracy"]

        # Get the run associated with the current production model
        prod_run_id = current_prod_model[0].run_id  # Get the run_id of the current production model
        prod_run = client.get_run(prod_run_id)  # Get the run associated with the production model

        # Get the current production model's accuracy from that run
        current_prod_accuracy = prod_run.data.metrics.get("accuracy", 0)

        logger.info(f"Latest Model Accuracy: {latest_accuracy}")
        logger.info(f"Current Production Model Accuracy: {current_prod_accuracy}")
    
        rounded_latest = round(latest_accuracy, 3)
        rounded_prod = round(current_prod_accuracy, 3)

        # Only promote if the latest accuracy is strictly greater than the production model's accuracy
        if rounded_latest >= rounded_prod:        
            logger.info("New model has higher accuracy. Promoting to production.")
            model_uri = f"runs:/{latest_run.run_id}/{model_name}"
            model_name = "HeartDiseaseModel"

            # Register and promote to production
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Production"
            )

            logger.info(f"Promoted model version {mv.version} with accuracy {latest_accuracy}")
        else:
            logger.info("Current production model has higher or equal accuracy. Not promoting.")
