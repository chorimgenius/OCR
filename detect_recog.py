import os, sys
import re
from clear_folder import clear_folder

current_folder_path = 'C:\\Users\\Admin\\Desktop\\ocr'

def extract_number(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return -1

def ocr(DT_model_path, DT_imagedir_path, RE_model_path):
    os.chdir(current_folder_path)
    
    # DETECTION
    terminnal_command1 = f'python  ./yolov5/detect.py --weight {DT_model_path} --img 640 --conf 0.4 --source {DT_imagedir_path} --save-crop --hide-labels --hide-conf' 
    os.system(terminnal_command1)
    
    # RECOGNITION을 하기 위한 디렉토리 변경
    print(os.getcwd())
    
    # RECOGNITION
    detect_path = 'yolov5/runs/detect'
    # detect_list
    detect_list = os.listdir(detect_path)
    sorted_detect_list = sorted(detect_list, key=extract_number)
    last_detect = sorted_detect_list[-1]
    crop_image_path = (f'{detect_path}/{last_detect}/crops')

    os.chdir('deep-text-recognition-benchmark')
    print(os.getcwd())
    
     # CUDA_VISIBLE_DEVICES=0    
    terminnal_command2 = f"""python demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--image_folder ../{crop_image_path} \
--saved_model  saved_models/TPS-ResNet-BiLSTM-CTC-Seed2022/{RE_model_path}
"""
    os.system(terminnal_command2)
    os.chdir(current_folder_path)
    clear_folder('image')
    
    
# ocr('./Easy-Yolo-OCR/yolov5/best.pt', './Easy-Yolo-OCR/yolov5/image', 'iter_10000.pth')

#   python ./detect.py --weight best.pt --img 640  --source ./image --device 0 --save-crop --hide-labels --hide-conf