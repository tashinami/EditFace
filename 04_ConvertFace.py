import os
import cv2
import glob
import shutil
import PIL
import PIL.Image
import numpy as np

from tqdm import tqdm
from itertools import chain

import dlib
from utils.alignment import get_landmark, get_align_info

kernel = np.ones((5,5),np.uint8)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def generate_face_mask(smile_image_name, ):
    # 笑顔画像のランドマークを取得
    smile_lm = get_landmark(smile_image_name, predictor)

    # 口周辺のマスク生成
    mouth_mask_points = []
    mouth_mask_points.append([smile_lm[29][0], smile_lm[29][1]])
    for point in smile_lm[1:15]:
        mouth_mask_points.append([point[0], point[1]])

    mouth_mask_points = np.array(mouth_mask_points)
    mouth_mask = np.zeros((smile_height, smile_width), np.uint8)
    mouth_mask = cv2.fillConvexPoly(mouth_mask, mouth_mask_points, color=(255, 255, 255))
    mouth_mask = cv2.erode(mouth_mask, kernel, iterations=10)
    mouth_mask = cv2.GaussianBlur(mouth_mask, (73, 73), 0)
    mouth_mask = cv2.GaussianBlur(mouth_mask, (73, 73), 0)
    mouth_mask = cv2.GaussianBlur(mouth_mask, (73, 73), 0)


    # 左目周辺
    left_eye_mask_points = []
    for point in smile_lm[36:41]:
        left_eye_mask_points.append([point[0], point[1]])

    # 右目周辺
    right_eye_mask_points = []
    for point in smile_lm[42:47]:
        right_eye_mask_points.append([point[0], point[1]])

    left_eye_mask_points = np.array(left_eye_mask_points)
    right_eye_mask_points = np.array(right_eye_mask_points)

    eyes_mask = np.zeros((smile_height, smile_width), np.uint8)
    eyes_mask = cv2.fillConvexPoly(eyes_mask, left_eye_mask_points, color=(255, 255, 255))
    eyes_mask = cv2.fillConvexPoly(eyes_mask, right_eye_mask_points, color=(255, 255, 255))
    eyes_mask = cv2.dilate(eyes_mask, kernel, iterations=2)
    eyes_mask = cv2.GaussianBlur(eyes_mask, (31, 31), 0)
    eyes_mask = cv2.GaussianBlur(eyes_mask, (31, 31), 0)
    eyes_mask = cv2.GaussianBlur(eyes_mask, (31, 31), 0)

    # 最終的なマスク画像
    mask_image = eyes_mask + mouth_mask

    return mask_image


if __name__ == "__main__":
    if os.path.isdir('results'):
        shutil.rmtree('results')
    os.makedirs('results', exist_ok=True)

    ext_list = ["jpg", "png", "jpeg"]
    image_list = sorted(list(chain.from_iterable([glob.glob(os.path.join("./images", "*." + ext)) for ext in ext_list])))

    for i, filepath in enumerate(tqdm(image_list)):
        # 元画像の読み込み
        img = PIL.Image.open(filepath)
        origin_image = cv2.imread(filepath)

        # ランドマーク検出 & クロップ情報の取得
        lm = get_landmark(filepath, predictor)
        quad, crop, pad = get_align_info(lm, img)

        # 笑顔変換した画像の読み込み
        smile_image_name = os.path.splitext(os.path.basename(filepath))[0]
        smile_image_name += ".png"
        smile_image_name = os.path.join("./smiled/", smile_image_name)
        smile_image = cv2.imread(smile_image_name)
        smile_height, smile_width = smile_image.shape[:2]

        # 射影変換行列を取得
        rect = [(int(point[0]), int(point[1])) for point in quad + 0.5]
        pts_src = [[0,0], [smile_width, 0], [smile_width, smile_height], [0, smile_height]]
        pts_dst = [[point[0] - pad[0] + crop[0], point[1] - pad[1] + crop[1]] for point in rect]
        perspective_mat = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))

        # 笑顔画像のマスク生成
        mask_image = generate_face_mask(smile_image_name)

        # マスク画像を参照してアルファ合成する
        smiled_origin_image = origin_image.copy()
        origin_height, origin_width = origin_image.shape[:2]
        is_inserted_mat = np.zeros((origin_height, origin_width), np.uint8)
        for y in range(smile_height):
            for x in range(smile_width):
                mask = mask_image[x, y]
                if mask > 0:
                    alpha = mask / 255.0
                    smile_pix = smile_image[x, y]

                    inv_xy = (
                        int(perspective_mat[0][0] * x + perspective_mat[0][1] * y + perspective_mat[0][2]),
                        int(perspective_mat[1][0] * x + perspective_mat[1][1] * y + perspective_mat[1][2])
                    )

                    origin_pix = origin_image[inv_xy[1], inv_xy[0]]
                    is_inserted = is_inserted_mat[inv_xy[1], inv_xy[0]]
                    if is_inserted == 0:
                        smiled_origin_image[inv_xy[1], inv_xy[0]] = smile_pix * alpha + origin_pix * (1.0 - alpha)
                        is_inserted_mat[inv_xy[1], inv_xy[0]] = 1
                    else:
                        inserted_pix = smiled_origin_image[inv_xy[1], inv_xy[0]]
                        smile_pix = smile_pix / 2 + inserted_pix / 2
                        smiled_origin_image[inv_xy[1], inv_xy[0]] = smile_pix * alpha + origin_pix * (1.0 - alpha)


        image_name = os.path.splitext(os.path.basename(filepath))[0]
        image_name += ".png"
        output_path = os.path.join("./results/", image_name)
        cv2.imwrite(output_path, smiled_origin_image)
