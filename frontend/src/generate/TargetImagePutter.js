import React, { useState } from "react";
import FabricCanvas from "./FabricCanvas";
import ImageUploader from "./ImageUploader";
import Tooltip from "./Tooltip";

const TargetImagePutter = ({
  setImageFile,
  setImage,
  images,
  image,
  page,
  activeTooltip,
}) => {
  const [imageWay, setImageWay] = useState("imageLoader");
  const onClick = (e) => {
    if (e.target.id === "canvas") {
      setImageWay("canvas");
    } else if (e.target.id === "imageLoader") {
      setImageWay("imageLoader");
    }
  };
  return (
    <div className="flex flex-col items-center">
      <div className="flex justify-between flex-row text-nowrap items-center w-96 justify-content-center mt-5">
        <Tooltip activeTooltip={activeTooltip} page={page}/>
        <div className="flex">
             <div onClick={onClick} className={` ${imageWay === "canvas"?'bg-darkblue2 text-greenblue':'bg-header-blue'}  h-10 p-2 text-center hover:cursor-pointer`} id="canvas">
          직접 그리기
        </div>
        <div
          onClick={onClick}
          className={`${imageWay === "imageLoader"?'bg-darkblue2 text-greenblue':'bg-header-blue'} h-10 p-2 text-center hover:cursor-pointer ml-3`}
          id="imageLoader"
        >
          이미지 올리기
        </div>
        </div>
       
      </div>
      {imageWay === "canvas" ? (
        <>
          <h1 className="text-2xl text-center mt-3">그림판</h1>
          <FabricCanvas setImageFile={setImageFile} setImage={setImage} />
        </>
      ) : imageWay === "imageLoader" ? (
        <>
          <h1 className="text-2xl text-center mt-3">이미지 업로더</h1>
          <ImageUploader
            images={images}
            image={image}
            setImage={setImage}
            setImageFile={setImageFile}
            page={page}
          />
        </>
      ) : (
        <></>
      )}
    </div>
  );
};

export default TargetImagePutter;
