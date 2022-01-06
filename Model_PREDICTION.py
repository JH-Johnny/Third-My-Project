
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
import json
import cv2
import os

# 학습 모델 예측(사용)하는 코드
def downscale(img, x):
    if x > 1280:
        print("Down Scale not available because you exceed maximum size")
        return
    y = int(0.75 * x)
    img = cv2.resize(img, (x, y))
    return img

with open("./source/Json2coco_dataset/Strange_Action_cctv.json", 'r', encoding="utf-8") as f:
    json_file = json.load(f)  # coco data_frame json

setup_logger()

jpg_path = "F:/지하철_역사_내_CCTV_이상행동_영상/Training/"
register_coco_instances("Action_Detecting", {}, "./source/Json2coco_dataset/Strange_Action_cctv.json", jpg_path)

person_metadata = MetadataCatalog.get("Action_Detecting")
dataset_dicts = DatasetCatalog.get("Action_Detecting")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Action_Detecting",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(json_file['categories'])

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("Action_Detecting",)
predictor = DefaultPredictor(cfg)

## 경로에 있는 이미지 다수를 차례로 읽어와 모델을 적용하여 저장하는 코드 --- (1)
# path = "F:/지하철_역사_내_CCTV_이상행동_영상/Validation/몰카/[원천]몰카_10/"
# file_list = os.listdir(path)
# inx = 0
# for i in file_list:
#     file_list2 = os.listdir(path+i)
#     for img in file_list2:
#         img_array = np.fromfile(path+i+"/"+img, np.uint8)  # 경로에 한글이 있으면 opencv2는 인식을 못하는 문제 해결
#         im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         # im = cv2.imread(img_array)
#         outputs = predictor(im)
#
#         # We can use `Visualizer` to draw the predictions on the image.
#         v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         cv2.imwrite("./result/"+str(inx)+".jpg", out.get_image()[:, :, ::-1])
#         inx += 1
## 코드 끝 -- (1)

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

cred_path = "./serviceAccountKey.json"
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# This registration token comes from the client FCM SDKs.
# Test용 Android 가상 기기의 토큰
# > 앱 다운 받은 기기의 토큰을 받아 저장하고 읽어와 코드에 사용하려면, 앱 출시에 대한 부분들 포함해서 추가적인 코딩을 해야함
registration_token = 'e0WkxMSaR9S0f3y2Hdh-hz:APA91bE7ThvpsUKmrDUO3QmVVVGxT9oqiP5eMpcPc29LE7g1alM4QxXMI4ab9SH7GNjEgJNUjJ49HlzE3Xq5Ulz4f7YOI6RyjhJ6adCAakhhq7tO4mIgHfRAVFRI36QxqNOZnYbuKL9P'
# APIKEY = "AAAAiJ4v7Lk:APA91bGWlLjzrM7sbuiHRRgM1qEiP5hGoTnQGdSNHcop7nAUHLnoOvMaPxea0tVguu7KmfAYRqvrITzpJz0z2bmIhMjDa1WH01uiNWSwlOo6Jt-BuTAv0L3Smm_lDZu3XhtUpp2x-ran"

# See documentation on defining a message payload.
message = messaging.Message(
notification=messaging.Notification(
    title='이상행동 감지',
    body='이상행동 확인이 필요합니다. 문제시 신고해주세요.',
),
token=registration_token
)

def check_in(a, b):

    for x in a:
        if x in b:
            return True
    return False

DB = []
DB2 = []
count = 0
file_list = os.listdir("./source/Video")
file_list = [file for file in file_list if file.endswith((".mkv", ".mp4", ".avi", ".wmv"))]

# flag = 1 # 이미지 옵션  0:흑백, 1:컬러, -1:투명도 채널 포함
if len(file_list) != 0:
    filename = file_list[0]

    video = cv2.VideoCapture("./source/Video/" + filename)
    width = int(video.get(3))
    height = int(video.get(4))
    fps = 20

    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # Mac ('M', 'J', 'P', 'G') Win ('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('output.mp4', fcc, fps, (width, height))

    while (video.isOpened()):
        ret, frame = video.read()  # ret은 프레임에 이미지를 불러왔을 때 성공여부, frame에 이미지로 저장됨
        if ret:
            # frame = cv2.flip(frame, 0)  # 이미지 상하 뒤짚음
            # out.write(frame)  # 위 코드에서 변경된 이미지 저장
            if int(video.get(1)) % 20 == 0:
                frame = downscale(frame, 1280)
                outputs = predictor(frame)
                accuracy = outputs['instances'].scores.tolist()
                class_predict =  outputs['instances'].pred_classes.tolist()
                if accuracy: # 감지한 객체정보 DB에 저장
                    if not DB:
                        for z in range(0, len(accuracy)):
                            if (round(accuracy[z], 2)*100) > 89:
                                DB.append(class_predict[z])
                    elif not DB2:
                        for z in range(0, len(accuracy)):
                            if (round(accuracy[z], 2)*100) > 89:
                                DB2.append(class_predict[z])
                else: # 아무것도 감지 못한경우 모든 DB 클리어
                    DB.clear()
                    DB2.clear()

                if DB: #연속적인 이상행동 감지 카운트
                    if DB2:
                        if check_in(DB, DB2):
                            if count%2==0:
                                count +=1
                                print("이상행동 연속 감지중! : ", count)
                                DB.clear()
                            else:
                                count +=1
                                print("이상행동 연속 감지중! : ", count)
                                DB2.clear()
                        else:
                            DB.clear()
                    else:
                        count =0
                else:
                    count =0

                if count==10: #연속으로 10장이상 이상행동 감지시 앱으로 알람 보냄
                    response = messaging.send(message)
                    print('Successfully sent message:', response)
                    count =0

                # We can use `Visualizer` to draw the predictions on the image.
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
                out2 = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imshow("Image of Frame", out2.get_image()[:, :, ::-1])
                # cv2.imshow('Image of Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            print("Fail to read frame!")
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
else:
    print("Oh Video is not exist!")