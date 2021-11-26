
# --- 顔画像の切り出し ---
import os
import glob
import shutil
from tqdm import tqdm
from itertools import chain

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

  ext_list = ["jpg", "png", "jpeg"]
  image_list = sorted(list(chain.from_iterable([glob.glob(os.path.join("./images", "*." + ext)) for ext in ext_list])))

  for i, file in enumerate(tqdm(image_list)):
    input_image = run_alignment(file)
    input_image.resize((256,256))

    image_name = os.path.splitext(os.path.basename(file))[0]
    image_name += ".png"
    output_path = os.path.join("./align/", image_name)
    input_image.save(output_path)
