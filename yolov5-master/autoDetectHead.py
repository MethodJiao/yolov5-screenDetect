import torch
import numpy as np
import cv2
from pathlib import Path
from mss import mss
import time
import math
import win32gui
import win32api
import win32con
import win32print
import time

is_release = True


def DrawRect(xmin, ymin, xmax, ymax):
    hwnd = win32gui.GetDesktopWindow()
    hPen = win32gui.CreatePen(win32con.PS_DASH, 1, win32api.RGB(255, 0, 0))  # 定义框颜色
    win32gui.InvalidateRect(hwnd, None, True)
    win32gui.UpdateWindow(hwnd)
    win32gui.RedrawWindow(
        hwnd,
        None,
        None,
        win32con.RDW_FRAME
        | win32con.RDW_INVALIDATE
        | win32con.RDW_UPDATENOW
        | win32con.RDW_ALLCHILDREN,
    )
    hwndDC = win32gui.GetDC(hwnd)  # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    win32gui.SelectObject(hwndDC, hPen)
    hbrush = win32gui.GetStockObject(win32con.NULL_BRUSH)  # 定义透明画刷
    prebrush = win32gui.SelectObject(hwndDC, hbrush)
    win32gui.Rectangle(
        hwndDC, round(xmin), round(ymin), round(xmax), round(ymax)
    )  # 左上到右下的坐标
    win32gui.SelectObject(hwndDC, hPen)


class screenpoint:
    def __init__(self, pt_x, pt_y) -> None:
        self.pt_x = pt_x
        self.pt_y = pt_y


def cal_distance(p1, p2):
    return math.sqrt(
        math.pow((p2.pt_x - p1.pt_x), 2) + math.pow((p2.pt_y - p1.pt_y), 2)
    )


def main_while_detect():
    if not is_release:
        if not cv2.getWindowProperty("detect", cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            exit("程序结束")

    screen_shot = screen_mss.grab(screen_size)
    screen_shot = np.array(screen_shot)

    results = model(screen_shot)

    if not is_release:
        results.display(render=True)
    if results.pandas().xyxy[0].shape[0] != 0:
        print(
            "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]",
            "person: ",
            results.pandas().xyxy[0].shape[0],
        )
    head_img = results.imgs[0]
    list_headpts = []
    for index, row in results.pandas().xyxy[0].iterrows():
        if len(row) != 7:
            continue
        xmin = row[0]
        xmax = row[2]
        ymin = row[1]
        ymax = row[3]

        if is_release:  # 去掉可提高性能
            DrawRect(xmin, ymin, xmax, ymax)

        head_point_x = xmin + ((xmax - xmin) / 2)
        head_point_y = ymin + ((ymax - ymin) * 0.1)
        list_headpts.append(screenpoint(head_point_x, head_point_y))
        if not is_release:
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
    cursor_pos = win32api.GetCursorPos()
    mouse_pt = screenpoint(cursor_pos[0], cursor_pos[1])
    dis_mouse = screen_size["width"]
    nearly_headpt = mouse_pt

    for x in list_headpts:
        temp_dis = cal_distance(x, mouse_pt)
        if temp_dis < dis_mouse:
            dis_mouse = temp_dis
            nearly_headpt = x

    if dis_mouse < adsorb_dis:
        win32api.SetCursorPos([round(nearly_headpt.pt_x), round(nearly_headpt.pt_y)])

    if not is_release:
        cv2.imshow("detect", head_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    hDC = win32gui.GetDC(0)
    screen_width = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)  # 横向分辨率
    screen_height = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)  # 纵向分辨率
    adsorb_dis = screen_height / 7

    FILE = Path(__file__).resolve()
    ROOT = str(FILE.parents[0])

    model = torch.hub.load(ROOT, "custom", path=ROOT + "\\yolov5s.pt", source="local")
    model.cuda()
    model.classes = [0]  # 只推理0标签 person
    screen_size = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
    if not is_release:
        screen_size = {"top": 0, "left": 0, "width": 800, "height": 640}
    screen_mss = mss()
    if not is_release:
        cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detect", 800, 640)
    previous_time = 0
    while True:
        main_while_detect()

        # fps_Num = "fps: %.1f" % (1.0 / (time.time() - previous_time))
        # previous_time = time.time()
        # print(fps_Num)
