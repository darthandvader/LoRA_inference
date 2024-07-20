import torch
import clip
from PIL import Image
import torch.nn.functional as F
import time
# from utils import *
from tqdm import tqdm
from inference_utils import *
import matplotlib.pyplot as plt


seed = 1  # if seed is needed
set_random_seed(seed)

args = get_arguments()

# Set the pretrained weight here, depending on which weight to use, end-effector or base.
args.weight_path = 'logs/vitb16/robotbase_test/32shots/seed1/lora_weights.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

list_lora_layers = apply_lora(args, model)
model.cuda()
load_lora(args, list_lora_layers)

# Enter inference img path.
img_path = 'DATA/base_test/images/of robot base/2023-09-25_25947356_frame_00015.jpg'
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
# Enter inference captions in the following format.
text = ["a photo without robot base", "a photo of robot base"]
text = clip.tokenize(text).cuda()
with torch.no_grad():
     with torch.cuda.amp.autocast():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 