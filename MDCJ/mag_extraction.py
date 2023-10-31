#!/usr/bin/env python

import os
import sys

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.0"
__date__ = "30/10/2023"

# Folder paths
# sorted_folder = r"D:\path\to\sorted_folder"
# extract_folder = r"D:\path\to\extract_folder"
# filter_dest = r"D:\path\to\filter_destination"

# File suffixes
suffix_sf = ".suffix_sorted_folder"
suffix_ef = ".suffix_folder_extract_folder"

# Fixed variables
# sorted_files = []
# px_overlap = "overlap in pixel"

# Variables for testing and debugging
sorted_folder = r"D:\PyCharm projects\ErasmusMC\patch_extraction\sorted_folder"
extract_folder = r"D:\PyCharm projects\ErasmusMC\patch_extraction\extract_folder"
filter_dest = r"D:\PyCharm projects\ErasmusMC\patch_extraction\filter_dest"

extract_files = []
sorted_files = []
px_overlap_pe = "overlap in pixel for patch_extraction"
px_overlap_sf = "overlap in pixel for sorted_folder"

# ADD SUFFIX VARIABLE
# INCREASE TESTING SIZE


if __name__ == "__main__":
    # Listing all files to extract
    for (dirpath, dirnames, filenames) in os.walk(extract_folder):
        extract_files.extend(filenames)

    # Walking through extract file
    for (dirpath, dirnames, filenames) in os.walk(sorted_folder):
        for i in range(0, len(filenames)):
            sort_file_id = filenames[i].split()[0]
            sort_file_cx = int(filenames[i].split(",")[1].replace("x=", ""))
            sort_file_cy = int(filenames[i].split(",")[2].replace("y=", ""))
            sort_file_cw = int(filenames[i].split(",")[3].replace("w=", ""))
            sort_file_ch = int(filenames[i].split(",")[4].replace("h=", "").replace("].txt", ""))
            sort_file_cx_end = sort_file_cx + sort_file_cw
            sort_file_cy_end = sort_file_cy - sort_file_ch
    print(sort_file_id,sort_file_cx,sort_file_cy,sort_file_cw,sort_file_ch,"\n")

    # Walking through each folder and listing all files
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

                        # print(extract_cx, extract_cx_end)
                        # print(extract_cy, extract_cy_end)

                        if extract_cx in range(sort_file_cx, sort_file_cx_end):
                            # print("MATCHED X-if:\t{dr}".format(dr=filenames[i]))
                            if extract_cy in range(sort_file_cy_end, sort_file_cy):
                                print("MATCHED XY-if:\t{dr}".format(dr=filenames[i]))
                            elif extract_cy_end in range(sort_file_cy_end, sort_file_cy):
                                print("MATCHED XY-elif:\t{dr}".format(dr=filenames[i]))
                        elif extract_cx_end in range(sort_file_cx, sort_file_cx_end):
                            # print("MATCHED X-elif:\t{dr}".format(dr=filenames[i]))
                            if extract_cy in range(sort_file_cy_end, sort_file_cy):
                                print("MATCHED XY-if:\t{dr}".format(dr=filenames[i]))
                            elif extract_cy_end in range(sort_file_cy_end, sort_file_cy):
                                print("MATCHED XY-elif:\t{dr}".format(dr=filenames[i]))
