import os
import shutil
import multiprocessing

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.0"
__date__ = "31/10/2023"

# Folder paths
sorted_folder = r"D:\path\to\sorted_folder"
extract_folder = r"D:\path\to\extract_folder"
filter_dest = r"D:\path\to\filter_destination"

# Variables
suffix_sort = "].txt"
suffix_extract = "].txt"
num_workers = 4


# Function to extract file
def extract(file_info, source_folder, dest_folder):
    file_id, cx, cy, cw, ch = file_info
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
                if (cx <= extract_cx <= cx + cw or cx <= extract_cx_end <= cx + cw) and \
                        (cy >= extract_cy >= cy - ch or cy >= extract_cy_end >= cy - ch):
                    source_path = os.path.join(root, filename)
                    dest_path = os.path.join(dest_folder, filename)
                    shutil.move(source_path, dest_path)
                    print("EXTRACTED:\t", filename)


def process_file(filename):
    if filename == "exclude_specific_file":
        return
    file_info = filename.split()[0], \
                int(filename.split(",")[1].replace("x=", "")), \
                int(filename.split(",")[2].replace("y=", "")), \
                int(filename.split(",")[3].replace("w=", "")), \
                int(filename.split(",")[4].replace("h=", "").replace(suffix_sort, ""))
    extract(file_info, extract_folder, filter_dest)


def main():
    sorted_files = os.listdir(sorted_folder)
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(process_file, sorted_files)
    pool.close()
    pool.join()
    print("[INFO]: Extraction completed")


if __name__ == "__main__":
    main()
