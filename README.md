```
# Load libraries
import torch
import torch.nn.functional as F
from voxlect.model.dialect.mms_dialect import MMSWrapper

# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
model = MMSWrapper.from_pretrained("tiantiaf/voxlect-indic-lid-mms-lid-256", cache_dir = "/home/ec2-user/SageMaker/hf_cache").to(device)
model.eval()

# Label List
dialect_list = [
    "as",
    "bn",
    "brx",
    "doi",
    "en",
    "gu",
    "hi",
    "kn",
    "ks",
    "kok",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
]
    
# Load data, here just zeros as the example
# Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
# So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
max_audio_length = 15 * 16000
data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]
logits, embeddings = model(data, return_feature=True)
    
# Probability and output
dialect_prob = F.softmax(logits, dim=1)
print(dialect_list[torch.argmax(dialect_prob).detach().cpu().item()])
```
