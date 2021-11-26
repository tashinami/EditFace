import os
import cv2
import gdown
import shutil
import numpy as np 
from PIL import Image
from zipfile import ZipFile

import clip

import torch
from MapTS import GetFs,GetBoundary,GetDt
from manipulate import Manipulator

pt_name = '003.pt' #@param {type:"string"}

if __name__ == "__main__":
    # 学習済みパラメータのダウンロード
    model_path = 'data_sc.zip'  
    if not os.path.exists(model_path):
        gdown.download('https://drive.google.com/u/1/uc?id=13CCGLcCw6_GMHe8cUBiaLlORzEK4gwso', model_path, quiet=False)
        with ZipFile("data_sc.zip", 'r') as zip:
            zip.extractall() 

    # CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # モデルに学習済みパラメータをロード
    M = Manipulator(dataset_name='ffhq')
    fs3 = np.load('./npy/ffhq/fs3.npy')
    np.set_printoptions(suppress=True)

    # --- 潜在変数の選択 ---
    pt_folder = 'vec/'
    latents=torch.load(pt_folder+pt_name)
    w_plus=latents.cpu().detach().numpy()
    M.dlatents=M.W2S(w_plus)

    M.num_images = 1
    M.alpha = [0]
    M.manipulate_layers = [0]
    codes,out = M.EditOneC(0,M.dlatents) 
    original = Image.fromarray(out[0,0]).resize((512,512))
    M.manipulate_layers = None

    # --- 編集テキスト入力 ---
    neutral='face' #@param {type:"string"}
    target='smiling face' #@param {type:"string"}
    classnames=[target,neutral]
    dt=GetDt(classnames,model)

    # --- alpha & beta の設定 ---
    beta = 0.1 #@param {type:"slider", min:0.08, max:0.3, step:0.01}

    os.makedirs('smiled_movie', exist_ok=True)
    image_name = os.path.splitext(os.path.basename(pt_name))[0]
    output_dir = os.path.join("./smiled_movie/", image_name)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.mp4")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1024, 1024))


    alpha_range = 3.0
    for count, alpha in enumerate(np.arange(0.0, alpha_range, 0.1)):
        M.alpha = [alpha]
        boundary_tmp2,c = GetBoundary(fs3,dt,M,threshold=beta)
        codes = M.MSCode(M.dlatents,boundary_tmp2)
        out = M.GenerateImg(codes)
        generated = Image.fromarray(out[0,0])#.resize((512,512))

        cv_image = np.array(generated)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        writer.write(cv_image)
