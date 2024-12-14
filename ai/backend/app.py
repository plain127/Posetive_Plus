from flask import Flask, request  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import boto3
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from omnipose import run_infer
from pe2pg import Pe2Pg
from pg2.tool import generate_pose_map_add_mask
from pg2 import run

load_dotenv()
app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

def s3_connection():
    
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!") 
        return s3

@api.route('/pgpg')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class PgpgGeneration(Resource):
    def post(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        data = request.get_json()
        generation_id = data.get("generationId")
        condition_image_url = data.get("conditionImageUrl")
        target_image_url = data.get("targetImageUrl")
        print(generation_id, condition_image_url, target_image_url)

        # HPE + PPE
        run_infer.run_infer(generation_id, condition_image_url, target_image_url)

        # mpii2coco + generate annotation, pairs
        p2g = Pe2Pg(generation_id)
        p2g.save_anno_csv()

        # generate posemap & mask
        generate_pose_map_add_mask.main("market", "inference", generation_id)

        # PG2
        run.main(generation_id)
        
        object_name = str(generation_id)+"_result.jpg"

        s3.upload_file("/model/generation/"+str(generation_id)+"/result/converted_condition_image.jpg___converted_target_image.jpg_vis.jpg", bucket_name, object_name, ExtraArgs={'ContentType': 'image/jpeg'})
        result_image_url = f"https://{bucket_name}.s3.{s3.meta.region_name}.amazonaws.com/{object_name}"
        
        return {"result": result_image_url}

if __name__ == "__main__":
    access_key = os.getenv('ACCESS_KEY')
    secret_key = os.getenv('SECRET_KEY')
    region_name = os.getenv('REGION_NAME')
    bucket_name = os.getenv('BUCKET_NAME')
    s3 = s3_connection()
    app.run(debug=True, host='0.0.0.0', port=5001)