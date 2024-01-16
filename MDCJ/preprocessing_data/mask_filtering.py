import os
import shutil
from PIL import Image

def move_image_with_structure(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(src, dest)

def has_excess_white_background(img_path, threshold):
    with Image.open(img_path) as img:
        pixels = img.convert('RGB').getdata()
        white_count = sum(1 for pixel in pixels if pixel == (255, 255, 255))
        total_pixels = img.width * img.height
        return (white_count / total_pixels) > threshold

def process_images(root_dir, target_dir, threshold):
    print(f"Starting processing for directory: {root_dir}")

    for foldername, subfolders, filenames in os.walk(root_dir):
        print(f"Processing folder: {foldername}")
        for filename in filenames:
            if filename.endswith('.png'):
                png_path = os.path.join(foldername, filename)
                print(f"Processing PNG: {png_path}")
                if has_excess_white_background(png_path, threshold):
                    print(f"PNG with excess white background: {png_path}")
                    jpg_filename = filename.replace('.png', '.jpg')
                    jpg_path = os.path.join(foldername, jpg_filename)
                    if os.path.exists(jpg_path):
                        print(f"Found corresponding JPG for PNG: {jpg_path}")
                        new_png_path = png_path.replace(root_dir, target_dir)
                        new_jpg_path = jpg_path.replace(root_dir, target_dir)
                        print(f"Moving {png_path} to {new_png_path}")
                        print(f"Moving {jpg_path} to {new_jpg_path}")
                        move_image_with_structure(png_path, new_png_path)
                        move_image_with_structure(jpg_path, new_jpg_path)
                    else:
                        print(f"No corresponding JPG found for PNG: {png_path}")
                else:
                    print(f"PNG does not exceed white background threshold: {png_path}")


if __name__ == "__main__":
    source_directory = r"D:\path\to\input\directory"
    target_directory = r"D:\path\to\output\directory"
    background_threshold = 0.75
    
    try:
        process_images(source_directory, target_directory, background_threshold)
    except Exception as e:
        print(f"Error encountered: {e}")
    print("Processing completed.")
