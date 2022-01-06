import os
import json

# 공공데이터 지하철 CCTV 이상행동 json 데이터를 coco data frame으로 변환하는 코드
# global variable
images = []
categories = []
annotations = []
img_id = []
def find_read_json(JSON_FILE):
    with open(JSON_FILE, 'r', encoding="utf-8") as f:
        json_file = json.load(f)
    # print(json.dumps(json_file, indent="\t")) # json파일을 들여쓰기 출력
    return json_file

def coco_frame_creat(json_f): # json_annotations is 'list'
    height = json_f['metadata']['height']
    width = json_f['metadata']['width']
    idx = len(categories)
    for loop in range(0, len(json_f['frames'])):
        filename = json_f['frames'][loop]['image']
        json_annotations = json_f['frames'][loop]['annotations']
        img_id.append(len(img_id))
        images.append(input_image(filename, height, width, len(img_id)-1))
        # 카테고리 id >> 카테고리에 속하는 id 값
        for i in json_annotations:
            categories_name_list = []
            for z in categories:
                categories_name_list.append(z['name'])
            bbox = list(i['label'].values())
            if i['category']['code'] in categories_name_list:
                annotations.append(input_annotation(0, len(img_id)-1, bbox, categories_name_list.index(i['category']['code']), len(annotations), bbox[2]*bbox[3]))
            else:
                categories.append(input_category(idx, i['category']['code']))
                annotations.append(input_annotation(0, len(img_id)-1, bbox, idx, len(annotations), bbox[2]*bbox[3]))
                idx += 1

        # {'images':images, 'categories':categories, 'annotations':annotations} 형식 완성
        # json_f['frames'][loop]['number']

    # # json 파일 생성 코드
    # with open("source/cctv_data/"+str(id)+"/frame_"+str(id)+".json", 'w', encoding='utf-8') as make_file:
    #     json.dump({'categories':categories, 'images':images, 'annotations':annotations}, make_file, ensure_ascii=False, indent="\t")

def input_image(file_name, height, width, id):
    return {"file_name": file_name, "height": height, "width":width, "id":id}

def input_category(id, name):
    return {"supercategory": "person", "id": int(id), "name": str(name)}

def input_annotation(iscrowd, image_id, bbox, category_id, id, area): # bbox and segmentation is list
    return {"iscrowd":iscrowd, "image_id":image_id, "bbox":bbox, "category_id":category_id, "id":id, "area":area}

# coco 데이터 변환... 단일 파일의 경우
# coco_frame_creat(find_read_json("./annotation_2248321.json"))

# coco 데이터 변환... 복수 파일의 경우
path = "./source/json_files/"
file_list = os.listdir(path)
for m in file_list:
    path = "./source/json_files/"+m+"/"
    file_list2 = os.listdir(path)
    file_list_json = [file for file in file_list2 if file.endswith(".json")]
    for n in file_list_json:
        print("파일명 :", n)
        coco_frame_creat(find_read_json(path+n))

with open("source/Json2coco_dataset/Strange_Action_cctv.json", 'w', encoding='utf-8') as make_file:
         json.dump({'categories':categories, 'images':images, 'annotations':annotations}, make_file, ensure_ascii=False, indent="\t")
print("COCO json File Create!")