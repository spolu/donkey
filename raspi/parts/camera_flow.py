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

        self.window = []

    def run(self, camera_raw = None):
        self.curr = cv2.imdecode(
            np.fromstring(camera_raw, np.uint8),
            cv2.CV_8UC1
        )[50:]

        dx = 0
        dy = 0
        norm = 0

        if self.prev is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev, self.curr,
                self.p0, None, **self.lk_params,
            )

            good_new = p1[st==1]
            good_old = self.p0[st==1]

            dxs = []
            dys = []
            norms = []

            for i, (new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()

                norm = (c-a)*(c-a)+(d-b)*(d-b)
                dx = c-a
                dy = d-b

                dxs += [dx]
                dys += [dy]
                if(dy < 0.5 and norm < 200):
                    norms += [norm]

            if len(dxs) > 0:
                dx = sum(dxs) / len(dxs)
                dy = sum(dys) / len(dys)

            norms = sorted(norms)
            if len(norms) > 0:
                norm = sum(norms)/len(norms)

        self.prev = self.curr

        self.window = [norm] + self.window
        self.window = self.window[:15]

        speed = int(sum(self.window)/len(self.window))

        print("({}, {}) {}".format(
            int(dx * 100),
            int(dy * 100),
            speed,
        ))

        return dx, dy, speed

