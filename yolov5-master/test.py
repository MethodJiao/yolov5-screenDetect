# import torch

# # 模型
# model = torch.hub.load('C:\\Users\\Method-PC\\Desktop\\yoloDemo\\yolov5-master', 'custom', path='C:\\Users\\Method-PC\\Desktop\\yoloDemo\\yolov5-master\\reson.pt', source='local')
# # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# # 图像
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # 推理
# results = model(img)

# # 结果
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.


import torch
import numpy as np
import cv2
from PIL import ImageGrab

model = torch.hub.load('C:\\Users\\Method-PC\\Desktop\\yoloDemo\\yolov5-master',\
     'custom', path='C:\\Users\\Method-PC\\Desktop\\yoloDemo\\yolov5-master\\reson.pt', source='local')
model.cuda()
# img = cv2.imread('C:\\Users\\Method-PC\\Desktop\\123.jpeg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# results = model(img)

# results.display(render=True)
# # results.show()
# cv2.imshow('window',cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB))


while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,0,800,640)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    results = model(screen)
    results.display(render=True)
    # final_result = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB)
    cv2.imshow('window',results.imgs[0])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break