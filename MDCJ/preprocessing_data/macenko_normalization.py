import os
import numpy as np
from PIL import Image


def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if is_image_file(filename):
                full_path = os.path.join(root, filename)
                image_files.append(full_path)
    return image_files


def generate_save_path(original_path, base_directory, base_save_dir):
    relative_path = os.path.relpath(original_path, base_directory)
    save_path = os.path.join(base_save_dir, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path


def get_arguments():
    class Args:
        imageFiles = get_image_files(r'D:\path\to\input\folder')
        referenceImage = r"D:\path\to\reference\image"
        saveNorm = 'yes'
        saveHE = 'no'
        Io = 255
        alpha = 1
        beta = 0.05

    return Args()


def get_ref_values(reference_img, Io=240, beta=0.15):
    h, w, c = reference_img.shape
    reference_img = reference_img.reshape((-1, 3))
    OD = -np.log((reference_img.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, 1)
    maxPhi = np.percentile(phi, 99)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

    return HE, maxC


def normalizeStaining(img, reference_img, saveNorm=None, saveHE=None, Io=240, alpha=1, beta=0.15):
    HERef, maxCRef = get_ref_values(reference_img, Io, beta)
    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log((img.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveNorm is not None:
        Image.fromarray(Inorm).save(saveNorm + '.jpg')

    if saveHE is not None:
        Image.fromarray(H).save(saveHE + '_H.jpg')
        Image.fromarray(E).save(saveHE + '_E.jpg')

    return Inorm, H, E


BASE_DIR = r'D:\path\to\input\directory'
SAVE_DIR = r'D:\path\to\output\directory'

if __name__ == '__main__':
    args = get_arguments()
    reference_img = np.array(Image.open(args.referenceImage))

    for image_file in args.imageFiles:
        if image_file.lower().endswith(('.jpg', '.jpeg')):
            img = np.array(Image.open(image_file))
            sid = image_file.split("_")[1]
            save_path_base = generate_save_path(image_file, BASE_DIR, SAVE_DIR)
            if args.saveNorm != 'no':
                save_path_norm = os.path.splitext(save_path_base)[0] + "_norm" + os.path.splitext(image_file)[1]
            else:
                save_path_norm = None
            if args.saveHE != 'no':
                save_path_he = os.path.splitext(save_path_base)[0] + "_HE"
            else:
                save_path_he = None
            normalizeStaining(img, reference_img, save_path_norm, save_path_he, args.Io, args.alpha, args.beta)
