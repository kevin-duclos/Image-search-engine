import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms


device = 'mps'
root = '/Users/kevinadmin/Desktop/Image Similarity/Oyster Larvae Training Set'
images = glob(os.path.join(root, '*.jpg'))
print(f'{len(images)} images')

os.environ["TORCH_HOME"] = "model/model_weights_edir"
model = torchvision.models.resnet18(weights="DEFAULT")
model.to(device)


all_names = []
all_vecs = None
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model.avgpool.register_forward_hook(get_activation("avgpool"))

# %%
with torch.no_grad():
    for i, file in enumerate(tqdm(images)):
        try:
            img = Image.open(file)
            img = transform(img)
            img = img.to(device)
            out = model(img[None, ...])
            vec = activation["avgpool"].cpu().numpy().squeeze()[None, ...]
            if all_vecs is None:
                all_vecs = vec
            else:
                all_vecs = np.vstack([all_vecs, vec])
            image_name = os.path.basename(file)
            all_names.append(image_name)
        except Exception as e:
            print(e)

        # if i % 100 == 0 and i != 0:
        #     print(i, "done")

# %% Save data
np.save(f"{root}/data/all_vecs.npy", all_vecs)
np.save(f"{root}/data/all_names.npy", all_names)
print('Exported data')
