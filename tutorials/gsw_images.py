# from https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html

import cv2 as cv
import sys

SAMPLES_PATH = '/home/miststlkr/.pyenv/versions/opencv/lib/python3.8/site-packages/cv2/samples/data/'
OUTPUT_PATH = './tutorials/outputs/'

if __name__ == '__main__':
    image_file: str = 'starry_night.{ext}'
    cv.samples.addSamplesDataSearchPath(SAMPLES_PATH)

    img = cv.imread(cv.samples.findFile(image_file.format(ext='jpg')))

    if img is None:
        sys.exit("Could not read the image.")

    cv.imshow("Display window", img)
    k = cv.waitKey(0)

    if k == ord("s"):
        cv.imwrite(OUTPUT_PATH + image_file.format(ext='png'), img)
