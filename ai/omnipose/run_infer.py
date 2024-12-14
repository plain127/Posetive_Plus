# 여기서 모델, 인풋, 아웃풋 경로 지정 (ppe, hpe)
# !python run_infer.py 만 실행시켜서 inference.py 돌리기
# run_infer.py 하나로 ppe랑 hpe가 한번씩 돌려서 각자 저장되게

import subprocess
import os
import requests

def download_image(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # URL이 유효하지 않을 경우 예외 발생

    with open(save_path, 'wb') as file:
        file.write(response.content)

def run_inference(model_file, image_path, output_dir, output_image_name, output_json_name):
    command = [
        'python', '/model/omnipose/inference.py', 
        '--model-file', model_file, 
        '--files-loc', image_path, 
        '--output-dir', output_dir,
        '--output-image-name', output_image_name,
        '--output-json-name', output_json_name
    ]
    subprocess.run(command, check=True)

def run_infer(generationId, conditionImageUrl, targetImageUrl):
    # 생성할 디렉토리 경로
    output_dir = f'/model/generation/{generationId}'  # 경로 수정
    os.makedirs(output_dir, exist_ok=True)

    # 다운로드할 파일 경로
    target_image_path = os.path.join(output_dir, 'target_image.jpg')
    condition_image_path = os.path.join(output_dir, 'condition_image.jpg')

    # URL에서 이미지를 다운로드하여 저장
    download_image(targetImageUrl, target_image_path)
    download_image(conditionImageUrl, condition_image_path)

    ppe_model_file = '/model/omnipose/checkpoint_ppe.pth'  # 경로 수정
    hpe_model_file = '/model/omnipose/checkpoint_hpe.pth'  # 경로 수정

    # PPE 인퍼런스
    run_inference(ppe_model_file, target_image_path, output_dir, 'converted_target_image.jpg', 'converted_target_pose.json')

    # HPE 인퍼런스
    run_inference(hpe_model_file, condition_image_path, output_dir, 'converted_condition_image.jpg', 'converted_condition_pose.json')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on multiple models')
    parser.add_argument('--generation-id', required=True, help='Generation ID for the inference run')
    parser.add_argument('--target-image-url', required=True, help='URL of the PPE input image')
    parser.add_argument('--condition-image-url', required=True, help='URL of the HPE input image')

    args = parser.parse_args()

    run_infer(args.generation_id, args.condition_image_url, args.target_image_url)
