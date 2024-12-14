import React, { useEffect, useState } from "react";
import Tooltip from "./Tooltip";
import ImageUploader from "./ImageUploader";
import arrow from "../asset/arrow.png";
import { useNavigate } from "react-router-dom";
import { getCookie } from "../components/Cookies";
import { axiosGenerationPgpg } from "../components/Axios";
import LoadingIndicator from "../components/LoadingIndicator";
import TargetImagePutter from "./TargetImagePutter";

const GenerateImage = () => {
  const navigate = useNavigate();
  useEffect(() => {
    if (!getCookie("accessToken")) navigate("/");
  }, []);
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [page, setPage] = useState(0);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [imageFiles, setImageFiles] = useState([]);

  useEffect(() => {
    const newImages = [...images];
    newImages[page] = image;
    setImages(newImages);
  }, [image]);
  useEffect(() => {
    const newImageFiles = [...imageFiles];
    newImageFiles[page] = imageFile;
    setImageFiles(newImageFiles);
  }, [imageFile]);

  const generateImage = async () => {
    if (!images[0] || !images[1]) {
      window.alert("이미지를 업로드해주세요!");
      return;
    }
    setLoading(true);
    setPage(2);
    axiosGenerationPgpg(imageFiles)
      .then((res) => {
        if (res.data.goojoCode === 201) {
          setImage(res.data.data.resultImageUrl);
          setLoading(false);
        } else {
          console.error(res.data.message);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };
  const previousPage = () => {
    if (page) setPage(page - 1);
  };

  const nextPage = () => {
    if (images[page] && page < 3) {
      setPage(page + 1);
    } else {
      window.alert("이미지를 업로드해주세요!");
    }
  };

  const title = ["1. 변경할 이미지 선택", "2. 원하는 자세 이미지 선택", "결과"];

  return (
    <div className="flex flex-col px-20 md:px-20 mt-8">
      <h1 className="text-3xl mb-9 text-center">{title[page]}</h1>
      {loading && <LoadingIndicator />}
      {page === 1 ? (
        <TargetImagePutter
          images={images}
          image={image}
          setImage={setImage}
          imageFile={imageFile}
          setImageFile={setImageFile}
          page={page}
        />
      ) : (
        <div className="flex flex-col items-center">
          {page != 2 && <div className="w-96"><Tooltip page={page}/></div>}
          <ImageUploader
            images={images}
            image={image}
            setImage={setImage}
            setImageFile={setImageFile}
            page={page}
          />
        </div>
      )}

      <div className="flex flex-row justify-between mt-5">
        <img
          src={arrow}
          className={`w-8 h-8 ${!page || page === 2 ? "invisible" : ""}`}
          onClick={previousPage}
        ></img>
        <img
          src={arrow}
          className={`w-8 h-8 rotate-180 ${page === 2 ? "invisible" : ""}`}
          onClick={page === 1 ? generateImage : nextPage}
        ></img>
      </div>
    </div>
  );
};

export default GenerateImage;
