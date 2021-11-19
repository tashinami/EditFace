


# # 5.ランドマークデータのダウンロード
# !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# !bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2


# --- 顔画像の切り出し ---
import os
import shutil
from tqdm import tqdm

import dlib
from utils.alignment import align_face

def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  return aligned_image 


if __name__ == "__main__":
  if os.path.isdir('align'):
      shutil.rmtree('align')
  os.makedirs('align', exist_ok=True)

  files = sorted(os.listdir('./images'))
  for i, file in enumerate(tqdm(files)):
    if file=='.ipynb_checkpoints':
      continue
    input_image = run_alignment('./images/'+file)
    input_image.resize((256,256))
    input_image.save('./align/'+file)
