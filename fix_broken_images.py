import os.path

import cv2 as cv

if __name__ == "__main__":
    path_file = 'E:\Programming\Python\Flet\CreaiveGAN\DATA\iconography'
    images_inside = [os.path.join(path_file, f) for f in os.listdir(path_file) if
                     os.path.exists(os.path.join(path_file, f))]
    for ii in images_inside:
        c = cv.imread(ii)
        if c is None:
            os.remove(ii)
            print(f'\033[1;36m{ii} Removed')
