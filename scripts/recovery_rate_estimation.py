import os

from recovery_rate_model import LSTMEstimator

relative_path_model_checkpoint = "artifacts/" + model_name + "/model.ckpt"
model_path = os.path.join(os.getcwd(), relative_path_model_checkpoint)

RR_Estimator = LSTMEstimator.load_from_checkpoint(model_path)