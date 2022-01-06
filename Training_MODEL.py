# Detectron2모델로 바운딩 박스 라벨링된 이미지를 학습하는 코드
def run():
    import torch
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data.catalog import DatasetCatalog
    import os
    import json
    from detectron2.data.datasets import register_coco_instances
    torch.multiprocessing.freeze_support()
    print('Run!')

    # # 각 디렉토리 별 학습
    # for n in file_list:
    #         file_list2 = os.listdir(path+n+"/") # source/cctv_data/xxxxxxx/~~ list -> .jpg 파일 위치 경로
    #         coco_data_json = [file for file in file_list2 if file.endswith(".json")] # source/cctv_data/xxxxxxx/frame_xxxx.json
    jpg_path = "F:/지하철_역사_내_CCTV_이상행동_영상/Training/"  # ./source/cctv_data/xxxxxxx/rame_xxxx.json
    register_coco_instances("Action_Detecting", {}, "./source/Json2coco_dataset/Strange_Action_cctv.json", jpg_path)
    with open("./source/Json2coco_dataset/Strange_Action_cctv.json", 'r', encoding="utf-8") as f:
        json_file = json.load(f) # coco data_frame json

    #vaildeation

    # metadata, dataca
    person_metadata = MetadataCatalog.get("Action_Detecting")
    dataset_dicts = DatasetCatalog.get("Action_Detecting")

    # training setting
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("Action_Detecting",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # Let training for Develop Model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 10000  # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, )
    cfg.SOLVER.GAMMA = 0.05

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # batch size 낮고 lr도 낮게 (16, 0.0001) / batch size 높고 lr도 높게 (256, 0.001)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(json_file['categories'])

    ## training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    run()

# 특정 .py 라던지 목록만 추출하는 코드
# file_list_json = [file for file in file_list if file.endswith(".json")]

# 각 파일 이미지 하나하나에 접근 code
# for n in file_list:
#     file_list2 = os.listdir(path+n+"/")
#     for m in file_list2:
#         cctv_data_list = os.listdir(path+str(m)+"/")
#         coco_json = [file for file in cctv_data_list if file.endswith(".json")]
#         for p in coco_json:
#             obj =  path+str(m)+p # ./source/cctv_data/xxxxxxx/rame_xxxx.json