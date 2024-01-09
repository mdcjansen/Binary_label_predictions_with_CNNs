#!/usr/bin/env python

import os
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.3"
__date__ = "08/11/2023"

path = r"\\path\to\input\folder"
shutil_path = r"\\path\to\output\folder"

identifier = set([])

if __name__ == "__main__":
    # finding and listing all unique identifiers
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            identifier.add(file.split()[0])
       
    # creating folders and moving files to said folder based on unique identifier
    for i in identifier:
        os.makedirs("{ph}\{ir}".format(ph=path,ir=i))
        files = [file.replace(".jpg.jpg", ".jpg") for file in os.listdir(path) if file.endswith(".jpg") if file.startswith(i)]
        for f in range(0, len(files)):
            print("{ph}\{ff}".format(ph=path,ff=files[f]), "{ph}\{ir}".format(ph=path,ir=i))
            shutil.move("{ph}\{ff}".format(ph=path,ff=files[f]), "{ph}\{ir}".format(ph=path,ir=i))
