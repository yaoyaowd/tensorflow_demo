import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from lane_detection import color_frame_pipeline


RESIZE_H = 540
RESIZE_W = 960
WORKING_DIR = '/Users/dwang/self-driving-car/'
verbose = True


if __name__ == '__main__':
    if verbose:
        plt.ion()

    test_images_dir = join(WORKING_DIR, 'project_1_lane_finding_basic/data/test_images/')
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

    for test_img in test_images:
        out_path = join(WORKING_DIR, 'out/images', basename(test_img))
        print "processing image: {} to {}".format(test_img, out_path)
        in_image = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out_image = color_frame_pipeline([in_image], solid_lines=True)
        cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
    plt.close('all')