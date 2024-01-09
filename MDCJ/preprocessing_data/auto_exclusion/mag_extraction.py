#!/usr/bin/env python

import os
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.3"
__date__ = "31/10/2023"

# Folder paths
sorted_folder = r"D:\path\to\sorted_folder"
extract_folder = r"D:\path\to\extract_folder"
filter_dest = r"D:\path\to\filter_destination"

# Variables
suffix_sort = ".suffix"
suffix_extract = ".suffix"


# Function to extract file
def extract(file_info, source_folder, dest_folder):
    file_id, cx, cy, cw, ch = file_info
    print(file_info)

    for root, dirs, filenames in os.walk(source_folder):
        for filename in filenames:
            if filename == "exclude_specific_file":
                continue
            if file_id in filename:
                extract_cx = int(filename.split(",")[1].replace("x=", ""))
                extract_cy = int(filename.split(",")[2].replace("y=", ""))
                extract_cw = int(filename.split(",")[3].replace("w=", ""))
                extract_ch = int(filename.split(",")[4].replace("h=", "").replace(suffix_extract, ""))
                extract_cx_end = extract_cx + extract_cw
                extract_cy_end = extract_cy - extract_ch
                if (extract_cx <= cx < extract_cx_end) and \
                    (extract_cx < cx + cw <= extract_cx_end):
                    if (extract_cy <= cy < extract_cy_end) and \
                            (extract_cy < cy + ch <= extract_cy_end):
                            source_path = os.path.join(root, filename)
                            dest_path = os.path.join(dest_folder, filename)
                            shutil.move(source_path, dest_path)
                            print("EXTRACTED:\t", filename)
                            sorted_files.remove(file_info)
                            print(len(sorted_files))


def main():
    sorted_files = os.listdir(sorted_folder)

    for sorted_filename in sorted_files:
        if sorted_filename == "exclude_specific_file":
            continue
        file_info = sorted_filename.split()[0], \
                    int(sorted_filename.split(",")[1].replace("x=", "")), \
                    int(sorted_filename.split(",")[2].replace("y=", "")), \
                    int(sorted_filename.split(",")[3].replace("w=", "")), \
                    int(sorted_filename.split(",")[4].replace("h=", "").replace(suffix_sort, ""))
        extract(file_info, extract_folder, filter_dest)

    print("[INFO]: Extraction completed")


if __name__ == "__main__":
    main()
