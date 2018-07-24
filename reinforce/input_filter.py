import math
import cv2
import random

import numpy as np

import reinforce

# import pdb; pdb.set_trace()

class InputFilter():
    def __init__(self, config):
        self.input_filter = config.get('input_filter')

    def shape(self):
        if self.input_filter == 'warp_line_detector':
            return (90, 160)
        return (
            int(reinforce.CAMERA_HEIGHT - 50),
            int(reinforce.CAMERA_WIDTH),
        )

    def apply(self, camera):
        img = camera[50:]

        if self.input_filter == 'warp_line_detector':
            pts1 = np.float32([
                [13, 33], [54, 8], [147, 33], [107, 8]
            ])
            pts2 = np.float32([
                [60, 80], [60, 20], [120, 80], [120, 20]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)

            h_base = np.array([
                -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1,
            ])
            arrays = [h_base for _ in range(12)]
            h_kernel = np.stack(arrays, axis=0)

            v_kernel = h_kernel.T

            d1_kernel = [
                [ 2,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [ 2,  2,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1,  2,  2,  2, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1,  2,  2,  2, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1,  2,  2,  2, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1,  2,  2,  2, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1,  2,  2,  2, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1,  2,  2,  2, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1,  2,  2,  2, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  2, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  2],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  2]
            ]
            d2_kernel = np.fliplr(d1_kernel)

            average_kernel = (
                v_kernel + h_kernel + d1_kernel + 0.25 + d2_kernel + 0.25
            ) / 4.0

            img = cv2.filter2D(
                cv2.warpPerspective(
                    img, M, (160,90),
                ),
                -1,
                average_kernel,
            )

            # Clamping is what gives us the very contrasted lines.
            img = np.clip(img, 0, 255.0)

        if self.input_filter == 'line_detector':
            h_base = np.array([-1]*4 + [2]*8 + [-1]*4)
            arrays = [h_base for _ in range(16)]
            h_kernel = np.stack(arrays, axis=0)

            v_kernel = h_kernel.T

            # TODO(matthieu): Create an array for a diagonal filter kernel.

            d1_array = [
                [-1] * max(i-3, 0) +
                [2] * min(i+3, 6, 19-i) +
                [-1] * max(13-i, 0)
                for i in range(16)
            ]
            d1_kernel = np.stack(d1_array, axis=0)
            d2_kernel = np.fliplr(d1_kernel)

            average_kernel = (
                v_kernel + h_kernel + d1_kernel + d2_kernel + 0.5
            ) / 2.0

            img = cv2.filter2D(img, -1, average_kernel)

        if self.input_filter == 'canny':
            img = cv2.Canny(np.uint8(img), 50, 150, apertureSize = 3,)

        return img
