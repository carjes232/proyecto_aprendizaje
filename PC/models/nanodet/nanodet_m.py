import torch
import logging
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.util import cfg

# Load the configuration file for NanoDet (replace with your config path)
cfg_path = 'nanodet-m-0.5x.yml'
cfg.merge_from_file(cfg_path)

# Build the NanoDet model
model = build_model(cfg.model)

# Create a logger
logger = logging.getLogger("NanoDet")
logger.setLevel(logging.INFO)  # Set the logging level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Load the weights from the .ckpt file
checkpoint = torch.load('nanodet_m_0.5x.ckpt', map_location='cpu')
load_model_weight(model, checkpoint, logger=logger)  # Pass the logger to the function

# Set the model to evaluation mode
model.eval()

# Save the model
torch.save(model.state_dict(), 'nanodet_m_0.5x.pt')

print("Model saved successfully as 'nanodet_m_0.5x.pt'")
