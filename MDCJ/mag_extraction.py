#!/usr/bin/env python

import os
import shutil
import sys

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.1"
__date__ = "31/10/2023"

# Folder paths
sorted_folder = r"D:\path\to\sorted_folder"
extract_folder = r"D:\path\to\extract_folder"
filter_dest = r"D:\path\to\filter_destination"

# Variables
sorted_files = []
suffix = "].txt"


# Function to identify file to sort
def sort_file(fileno):
    global sort_file_id, sort_file_cx, sort_file_cy, sort_file_cw, sort_file_ch, sort_file_cx_end, sort_file_cy_end
    for (dirpath, dirnames, filenames) in os.walk(sorted_folder):
        sort_file_id = filenames[fileno].split()[0]
        sort_file_cx = int(filenames[fileno].split(",")[1].replace("x=", ""))
        sort_file_cy = int(filenames[fileno].split(",")[2].replace("y=", ""))
        sort_file_cw = int(filenames[fileno].split(",")[3].replace("w=", ""))
        sort_file_ch = int(filenames[fileno].split(",")[4].replace("h=", "").replace(suffix, ""))
        sort_file_cx_end = sort_file_cx + sort_file_cw
        sort_file_cy_end = sort_file_cy - sort_file_ch


# Function to extract file
def extract(cx, cy, cx_end, cy_end):
    if cx in range(sort_file_cx, sort_file_cx_end):
        if cy in range(sort_file_cy_end, sort_file_cy):
            shutil.move("{dp}\{fi}".format(dp=dirpath, fi=filenames[i]), filter_dest)
            print("EXTRACTED:\t{dr}".format(dr=filenames[i]))
        elif cy_end in range(sort_file_cy_end, sort_file_cy):
            shutil.move("{dp}\{fi}".format(dp=dirpath, fi=filenames[i]), filter_dest)
            print("EXTRACTED:\t{dr}".format(dr=filenames[i]))
    elif cx_end in range(sort_file_cx, sort_file_cx_end):
        if cy in range(sort_file_cy_end, sort_file_cy):
            shutil.move("{dp}\{fi}".format(dp=dirpath, fi=filenames[i]), filter_dest)
            print("EXTRACTED:\t{dr}".format(dr=filenames[i]))
        elif cy_end in range(sort_file_cy_end, sort_file_cy):
            shutil.move("{dp}\{fi}".format(dp=dirpath, fi=filenames[i]), filter_dest)
            print("EXTRACTED:\t{dr}".format(dr=filenames[i]))


if __name__ == "__main__":

    # Listing all files from sorted folder
    for (dirpath, dirnames, filenames) in os.walk(sorted_folder):
        sorted_files.extend(filenames)

    # Extracting files
    for f in range(0, len(sorted_files)):
        sort_file(f)
        for (dirpath, dirnames, filenames) in os.walk(extract_folder):
            if len(dirnames) == 0:
                for i in range(0, len(filenames)):
                    if filenames[i].split()[0] in sort_file_id:
                        if filenames[i] != "exclude_specific_file":
                            extract_cx = int(filenames[i].split(",")[1].replace("x=", ""))
                            extract_cy = int(filenames[i].split(",")[2].replace("y=", ""))
                            extract_cw = int(filenames[i].split(",")[3].replace("w=", ""))
                            extract_ch = int(filenames[i].split(",")[4].replace("h=", "").replace("].txt", ""))
                            extract_cx_end = extract_cx + extract_cw
                            extract_cy_end = extract_cy - extract_ch
                            extract(extract_cx, extract_cy, extract_cx_end, extract_cy_end)
    print("[INFO]:\t\tExtraction completed")
    sys.exit(0)
