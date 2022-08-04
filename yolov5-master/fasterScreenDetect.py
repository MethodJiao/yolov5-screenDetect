import torch
import numpy as np
import cv2
from pathlib import Path
from mss import mss
import time

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0])  # YOLOv5 root directory

# 本地模型加载 reson.pt 换成 yolov5s.pt也行
model = torch.hub.load(ROOT, "custom", path=ROOT + "\\yolov5s.pt", source="local")

# 替换上一行本地模型加载为网络加载yolov5s (yolov5s可以写yolov5n - yolov5s - yolov5m - yolov5l - yolov5x)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# model.cuda() # 指定GPU运行

screen_size = {"top": 0, "left": 0, "width": 800, "height": 640}
screen_mss = mss()

cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("detect", 800, 640)
previous_time = 0
while True:
    # 判是否点击关闭按钮
    if not cv2.getWindowProperty("detect", cv2.WND_PROP_VISIBLE):
        cv2.destroyAllWindows()
        exit("程序结束")
    # mss快速捕获
    frame = screen_mss.grab(screen_size)
    frame = np.array(frame)
    # 推理
    results = model(frame)
    results.display(render=True)
    # 展示窗口并赋予推理后图片
    cv2.imshow("detect", results.imgs[0])
    cv2.waitKey(1)
    # 帧率显示
    fps_Num = "fps: %.1f" % (1.0 / (time.time() - previous_time))
    previous_time = time.time()
    print(fps_Num)
