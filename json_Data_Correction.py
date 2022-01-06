import os
import json
# json 파일 또는 coco dataset 데이터 수정을 위한 코드
path = "./source/json_files/"
file_list = os.listdir(path)
person_list = ["child", "wheelchair", "merchant", "blind", "stroller", "drunk", "public_intoxication"]
fall_list = ["fall"]

def correct_json(): # 원천데이터 json파일 수정하는 함수
    for m in file_list:
        path = "./source/json_files/"+m+"/"
        file_list2 = os.listdir(path)
        file_list_json = [file for file in file_list2 if file.endswith(".json")]
        for n in file_list_json:
            print("파일명 :", n)
            with open(path+n, 'r', encoding="utf-8") as f:
                json_file = json.load(f)

            ID = json_file['id']
            for i in range(0, len(json_file['frames'])):
                # filename = str(ID)+"_"+json_file['frames'][i]['image']
                json_annotations = json_file['frames'][i]['annotations']
                # json_file['frames'][i]['image'] = filename
                # 카테고리 id >> 카테고리에 속하는 id 값
                # for j in json_annotations:
                #     bbox = list(j['label'].values())
                #     if j['category']['code'] in person_list:
                #         j['category']['code'] = "person"
                #     elif j['category']['code'] == "turnstile_trespassing":
                #         j['category']['code'] = "trespassing"
                #     elif j['category']['code'] == "wrong_passing":
                #         j['category']['code'] = "person"
                #     elif j['category']['code'] == "property_damage":
                #         j['category']['code'] = "damage"
                #     elif j['category']['code'] in fall_list:
                #         j['category']['code'] = "person"
            with open(path+n, 'w', encoding='utf-8') as make_file:
                json.dump(json_file, make_file, ensure_ascii=False, indent="\t")
    print("모든 json 파일 수정 완료!")
# 목표 categoies : person, trespassing, wrong_passing, fall, damage, wandering, spy_camera, assault, fainting, theft, unattended
# 11종류

def correct_coco_json(): # 가공된 최종 cocodataset을 수정하는 함수
    path = "./source/Json2coco_dataset/Strange_Action_cctv.json"
    delete_list = ["child", "merchant", "wheelchair", "blind", "stroller", "drunk", "escalator_fall", "turnstile_wrong_direction", "wandering",
                   "surrounding_fall", "public_intoxication", "unattended"]
    category_id = []
    with open(path, 'r', encoding="utf-8") as f:
        coco_file = json.load(f)
    inx = 0
    A = []
    for i in coco_file['categories']:
        if i['name'] in delete_list:
            category_id.append(coco_file['categories'][inx]['id'])
        else:
            A.append(i)
        inx += 1
    coco_file['categories'] = A

    A = []
    for j in coco_file['annotations']:
        if j["category_id"] in category_id:
            print()
        else:
            A.append(j)
    coco_file['annotations'] = A

    with open(path, 'w', encoding='utf-8') as make_file:
        json.dump(coco_file, make_file, ensure_ascii=False, indent="\t")
    print("coco dataset 수정 완료!")

correct_coco_json()