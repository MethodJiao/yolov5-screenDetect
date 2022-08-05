import torch
import numpy as np
import cv2
from pathlib import Path
from mss import mss
import time
import pandas
import pyautogui
import math


class screenpoint:
    def __init__(self, pt_x, pt_y) -> None:
        self.pt_x = pt_x
        self.pt_y = pt_y


def cal_distance(p1, p2):
    return math.sqrt(
        math.pow((p2.pt_x - p1.pt_x), 2) + math.pow((p2.pt_y - p1.pt_y), 2)
    )


if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = str(FILE.parents[0])

    model = torch.hub.load(ROOT, "custom", path=ROOT + "\\reson.pt", source="local")
    model.cuda()
    model.classes = [0]  # 只推理0标签 person

    screen_size = {"top": 0, "left": 0, "width": 800, "height": 640}
    screen_mss = mss()

    cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detect", 800, 640)
    # previous_time = 0
    while True:
        if not cv2.getWindowProperty("detect", cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            exit("程序结束")

        frame = screen_mss.grab(screen_size)
        frame = np.array(frame)

        results = model(frame)
        results.display(render=True)
        # print(results.pandas().xyxy[0])
        head_img = results.imgs[0]
        list_headpts = []
        for index, row in results.pandas().xyxy[0].iterrows():
            if len(row) != 7:
                continue
            xmin = row[0]
            xmax = row[2]
            ymin = row[1]
            ymax = row[3]

            # print(
            #     "index : ",
            #     index,
            #     "xmin : ",
            #     xmin,
            #     "xmax : ",
            #     xmax,
            #     "ymin : ",
            #     ymin,
            #     "ymax : ",
            #     ymax,
            # )

            head_point_x = xmin + ((xmax - xmin) / 2)
            head_point_y = ymin + ((ymax - ymin) * 0.1)
            list_headpts.append(screenpoint(head_point_x, head_point_y))
            head_img = cv2.circle(
                head_img,
                center=(
                    round(head_point_x),
                    round(head_point_y),
                ),
                radius=5,
                color=(255, 0, 3),
                thickness=6,
            )

        x, y = pyautogui.position()
        mouse_pt = screenpoint(x, y)
        dis_mouse = 1920
        nearly_headpt = mouse_pt
        for x in list_headpts:
            temp_dis = cal_distance(x, mouse_pt)
            if temp_dis < dis_mouse:
                dis_mouse = temp_dis
                nearly_headpt = x
        if dis_mouse < 100:
            pyautogui.moveTo(nearly_headpt.pt_x, nearly_headpt.pt_y)
        cv2.imshow("detect", head_img)
        cv2.waitKey(1)

        # fps_Num = "fps: %.1f" % (1.0 / (time.time() - previous_time))
        # previous_time = time.time()
        # print(fps_Num)
