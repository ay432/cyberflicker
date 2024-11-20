# GAN
import os
import cv2
import numpy as np
from deepfake_face_detection import select_face
from deepfake_face_swap import (warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colors,
                                transformation_from_points, ProcessFace)

source_image_path = ""
dest_image_path = ""
out_image_path = ""

src_image = cv2.imread(source_image_path)
dst_image = cv2.imread(dest_image_path)

src_points, src_shape, src_face = select_face(src_image)
dst_points, dst_shape, dst_face = select_face(dst_image)

output = ProcessFace(src_points, src_face, dst_points, dst_face)

x, y, w, h = dst.shape
dst_img_cp = dst_img.copy()
dst_img_cp[y:y+h, x:x+w] = output
output = dst_img_cp
cv2.imwrite(out_image_path, output)


