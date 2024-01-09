#!/usr/bin/env python

import os
import sys
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.5"
__date__ = "27/10/2023"

# Folder paths
sorted_folder = r"D:\path\to\sorted_folder"
folder_to_sort = r"D:\path\to\folder_to_sort"
destination = r"D:\path\to\destination_folder"

# File suffixes
suffix_sf = ".suffix_sorted_folder"
suffix_fts = ".suffix_folder_to_sort"

# List of files
sorted_files = []

if __name__ == "__main__":

    # Checking for destination folder and producing it if it does not exist
    if not os.path.exists(destination):
        os.makedirs(destination)

    # listing all previously sorted files
    for (dirpath, dirnames, filenames) in os.walk(sorted_folder):
        sorted_files.extend(filenames)
    sorted_files_sc = [suffix.replace(suffix_sf, suffix_fts) for suffix in sorted_files]

    # finding and moving previously filtered files
    for (dirpath, dirnames, filenames) in os.walk(folder_to_sort):
        if len(dirnames) == 0:
            for i in range(0, len(filenames)):
                if filenames[i] in sorted_files_sc:
                    if filenames[i] != "exclude_specific_file":
                        print("MATCHEDFILE:\t{dp}\{dr}".format(dp=dirpath, dr=filenames[i]))
                        shutil.move("{dp}\{dr}".format(dp=dirpath, dr=filenames[i]), destination)
    sys.exit(0)
