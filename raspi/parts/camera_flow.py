import time
import cv2

import numpy as np

class CameraFlow:
    def __init__(self):
        self.lk_params = dict(
            winSize  = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.prev = None
        self.curr = None

        self.p0 = np.array([
            [[50,10]], [[75,10]], [[100, 10]],
            [[50,20]], [[75,20]], [[100, 20]],
        ], dtype=np.float32)

    def run(self, camera_raw = None):
        self.curr = cv2.imdecode(
            np.fromstring(camera_raw, np.uint8),
            cv2.CV_8UC1
        )[50:]

        dx = 0
        dy = 0

        if self.prev != None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev, self.curr,
                p0, None, **self.lk_params,
            )

            good_new = p1[st==1]
            good_old = p0[st==1]

            for i, (new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                dx += c-a
                dy += d-b

            dx /= len(good_new)
            dy /= len(good_new)

        self.prev = self.curr

        print("(" + str(dx), "," + str(dy) + ")")

        return dx, dy

