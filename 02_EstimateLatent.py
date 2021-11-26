import os
import gdown
import shutil
from PIL import Image
from tqdm import tqdm
from argparse import Namespace

from models.psp import pSp
from utils.common import tensor2im

import torch
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    if os.path.isdir('vec_pic'):
        shutil.rmtree('vec_pic')
    os.makedirs('vec_pic', exist_ok=True)

    if os.path.isdir('vec'):
        shutil.rmtree('vec')
    os.makedirs('vec', exist_ok=True)

    # モデルのダウンロード
    model_path = 'pretrained_models/e4e_ffhq_encode.pt'  
    os.makedirs('pretrained_models', exist_ok=True)
    if not os.path.exists(model_path):
        gdown.download('https://drive.google.com/u/1/uc?id=1Du_8FzOPKJhk6aJmiOBhAWVe3_6vAyET', model_path, quiet=False)

    # モデルの読み込み
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts).to(device)
    net.eval()
    print('Model successfully loaded!')

    # 前処理
    img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    files = sorted(os.listdir('./align'))
    for i, file in enumerate(tqdm(files)):
        if file=='.ipynb_checkpoints':
            continue
        input_image = Image.open('./align/'+file)
        transformed_image = img_transforms(input_image)

        with torch.no_grad():
            # 潜在変数の推定
            images, latents = net(transformed_image.unsqueeze(0).to(device).float(), randomize_noise=False, return_latents=True)
            result_image, latent = images[0], latents[0]

            # 保存
            tensor2im(result_image).save('./vec_pic/'+file) 
            torch.save(latents, './vec/'+file[:-4]+'.pt') 