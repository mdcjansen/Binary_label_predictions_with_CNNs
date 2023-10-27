#!/usr/bin/env python

import os
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.3"
__date__ = "26/10/2023"

# paths
sorted_folder = r"D:\PyCharm projects\ErasmusMC\sorted_files"
folder_to_sort = r"D:\PyCharm projects\ErasmusMC\sort_folder"
destination = r"D:\PyCharm projects\ErasmusMC\sort_folder"

sorted_files = []

if __name__ == "__main__":

    # Checking for destination folder and producing it if it does not exist
    if not os.path.exists(destination):
        os.makedirs(destination)

    # listing all previously sorted files
    for (dirpath, dirnames, filenames) in os.walk(sorted_folder):
        sorted_files.extend(filenames)
    sorted_files_sc = [suffix.replace(".png.txt", ".txt.txt") for suffix in sorted_files]

    # print("SORTED_FILES:\t\t", sorted_files)
    # print("SORTED_FILES_SC:\t", sorted_files_sc, "\n")

    # finding and moving previously filtered files
    for (dirpath, dirnames, filenames) in os.walk(folder_to_sort):
        if len(dirnames) == 0:
            # print("DIRPATH:\t\t", dirpath)
            # print("DIRFILES:\t\t", filenames)
            for i in range(0, len(filenames)):
                if filenames[i] in sorted_files_sc:
                    if filenames[i] != "exclude_specific_file":
                        print("MATCHEDFILE:\t {dp}\{dr}".format(dp=dirpath, dr=filenames[i]))
                        # shutil.copy("{dp}\{dr}".format(dp=dirpath, dr=filenames[i]), destination)
                        # shutil.move("{dp}\{dr}".format(dp=dirpath, dr=filenames[i]), destination)
            print("")
