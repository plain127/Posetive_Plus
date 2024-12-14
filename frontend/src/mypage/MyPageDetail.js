import React, { useEffect, useState } from "react";
import { axiosMyPageDetail } from "../components/Axios";
import { getCookie } from "../components/Cookies";
import { useNavigate, useParams } from "react-router-dom";

const MyPageDetail = () => {
  let { imageId } = useParams();
  const [conditionImageUrl, setConditionalImageUrl] = useState(null);
  const [targetImageUrl, setTargetImageUrl] = useState(null);
  const [resultImageUrl, setResultImageUrl] = useState(null);
  const navigate = useNavigate();
  useEffect(() => {
    if (!getCookie("accessToken")) {
      navigate("/");
    }
    getMyPageDetail();
  }, []);
  const getMyPageDetail = async () => {
    axiosMyPageDetail(imageId)
      .then((res) => {
        const data = res.data.data;
        setConditionalImageUrl(data.conditionImageUrl);
        setTargetImageUrl(data.targetImageUrl);
        setResultImageUrl(data.resultImageUrl);
      })
      .catch((err) => {
        console.error(err);
      });
  };
  return (
    <div className="container mx-auto px-4 flex flex-col pt-24 items-center min-h-screen w-full mb-20">
      <div className="flex flex-row">
        <div className="flex flex-col items-center flex-shrink-0">
          <div>포즈를 바꿀 이미지</div>
          <img
            alt="condtion-image"
            src={conditionImageUrl}
            className="max-w-52"
          ></img>
        </div>
        <div className="flex flex-col items-center">
          <div>포즈 픽토그램</div>
          <img
            alt="target-image"
            src={targetImageUrl}
            className="max-w-52"
          ></img>
        </div>
      </div>
      <div className="flex flex-col items-center flex-shrink-0">
        <div>결과 이미지</div>
        <img alt="result-image" src={resultImageUrl}></img>
      </div>
    </div>
  );
};

export default MyPageDetail;
