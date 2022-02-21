import os

from recovery_rate_model import LSTMEstimator
import wandb

model_name = "model-2616nj7s:v28"
DOWNLOAD_MODEL = True
if DOWNLOAD_MODEL:
    #connect to wnadb api.
    api = wandb.Api()
    latest_model = api.artifact('aksoym/recovery_rate/' + model_name)
    latest_model.download()


relative_path_model_checkpoint = "artifacts/" + model_name + "/model.ckpt"
model_path = os.path.join(os.getcwd(), relative_path_model_checkpoint)

RR_Estimator = LSTMEstimator.load_from_checkpoint(model_path)

