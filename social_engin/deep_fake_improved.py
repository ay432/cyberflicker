import os
import cv2
import numpy as np
from deepfake_face_detection import select_face
from deepfake_face_swap import (
    warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colors,
    transformation_from_points, ProcessFace
)

# https://github.com/emmanueltsukerman/deepface

def load_image(image_path):
    """Load an image from the given path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2.imread(image_path)


def save_image(image, output_path):
    """Save the processed image to the output path."""
    cv2.imwrite(output_path, image)


def extract_face(image):
    """Extract face points and shape from the image."""
    points, shape, face = select_face(image)
    return points, shape, face


def apply_face_swap(src_image, dst_image, src_points, src_face, dst_points, dst_face):
    """Apply the face swap transformation using the extracted faces."""
    output = ProcessFace(src_points, src_face, dst_points, dst_face)
    return output


def blend_faces(dst_image, output, dst_points):
    """Blend the swapped face into the destination image."""
    x, y, w, h = dst_points
    dst_image_copy = dst_image.copy()  # Make a copy to avoid modifying the original
    dst_image_copy[y:y + h, x:x + w] = output
    return dst_image_copy


def main(source_image_path, dest_image_path, out_image_path):
    """Main function to perform face swapping."""
    try:
        # Load source and destination images
        src_image = load_image(source_image_path)
        dst_image = load_image(dest_image_path)

        # Extract faces from both images
        src_points, _, src_face = extract_face(src_image)
        dst_points, _, dst_face = extract_face(dst_image)

        # Apply face swap
        output = apply_face_swap(src_image, dst_image, src_points, src_face, dst_points, dst_face)

        # Blend the face swap result into the destination image
        final_output = blend_faces(dst_image, output, dst_points)

        # Save the output image
        save_image(final_output, out_image_path)
        print(f"Face swap completed successfully. Output saved to {out_image_path}")

    except Exception as e:
        print(f"Error during face swapping: {e}")


# Example usage
source_image_path = "path/to/source/image.jpg"
dest_image_path = "path/to/destination/image.jpg"
out_image_path = "path/to/output/image.jpg"

main(source_image_path, dest_image_path, out_image_path)