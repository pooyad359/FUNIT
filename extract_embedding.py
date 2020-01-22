import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from utils import get_config
from trainer import Trainer
import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path','-p',type = str, required= True, help = 'Path to folder containing images.')
parser.add_argument('--name','-n',type = str, default=None, help = 'Name of the output pickle file. e.g. tiger.pk')

args = parser.parse_args()
path = os.path.normpath(args.path)
assert os.path.isdir(path), 'Path must be directory.'
if args.name is None:
    name = os.path.basename(path)
    filename = name + '.pk'
else:
    filename = args.name

INPUT_SIZE = 256

config = get_config('configs/funit_animals.yaml')
config['batch_size'] = 1
trainer = Trainer(config)
trainer.load_ckpt('pretrained/animal149_gen.pt')
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))] + transform_list
transform = transforms.Compose(transform_list)
class_image_folder = path
# print('Compute average class codes for images in %s' % class_image_folder)
images = os.listdir(class_image_folder)
for i, f in tqdm(enumerate(images)):
    fn = os.path.join(class_image_folder, f)
    img = Image.open(fn).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        class_code = trainer.model.compute_k_style(img_tensor, 1)
        if i == 0:
            new_class_code = class_code
        else:
            new_class_code += class_code
final_class_code = new_class_code / len(images)
print(f'saving the embeddings to {filename}')
with open(filename,'wb') as fp:
    pickle.dump(final_class_code,fp)