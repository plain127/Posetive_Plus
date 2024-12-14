import React, { useEffect, useRef, useState } from "react";
import ResultMenuBar from "./ResultMenuBar";

const ImageUploader = ({ images, image, setImage, setImageFile, page }) => {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {}, [image]);
  const renderFile = (file) => {
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
      setImage(reader.result);
    };
  };
  const checkFileSize = (file) => {
    let maxSize = 4 * 1024 * 1024;
    let fileSize = file.size;
    if (fileSize > maxSize) {
      alert("첨부파일 사이즈는 4MB 이내로 등록 가능합니다.");
      return false;
    }
    return true;
  };
  const handleDrop = (e) => {
    if (page == 2) return;
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (e.dataTransfer.items) {
      if (e.dataTransfer.items.length > 1) {
        alert("사진은 하나만 업로드 가능합니다.");
        return;
      } else if (
        e.dataTransfer.items[0].getAsFile().type !== "image/jpeg" &&
        e.dataTransfer.items[0].getAsFile().type !== "image/png" &&
        e.dataTransfer.items[0].getAsFile().type !== "image/gif"
      ) {
        alert("이미지 파일만 업로드 가능합니다.");
        return;
      } else if (checkFileSize(e.dataTransfer.items[0].getAsFile())) {
        renderFile(e.dataTransfer.items[0].getAsFile());
        setImageFile(e.dataTransfer.items[0].getAsFile());
      }
    } else {
      if (e.dataTransfer.files.length > 1) {
        alert("사진은 하나만 업로드 가능합니다.");
        return;
      } else if (
        e.dataTransfer.files[0].type !== "image/png" &&
        e.dataTransfer.files[0].type !== "image/jpeg" &&
        e.dataTransfer.files[0].type !== "image/gif"
      ) {
        alert("이미지 파일만 업로드 가능합니다.");
        return;
      } else if (checkFileSize(e.dataTransfer.files[0])) {
        renderFile(e.dataTransfer.files[0]);
        setImageFile(e.dataTransfer.files[0]);
      }
    }
  };
  const handleDragOver = (e) => {
    if (page == 2) return;
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  };
  const handleDragleave = (e) => {
    setDragOver(false);
  };
  const handleFile = (e) => {
    if (page == 2) return;
    if (e.target.files && e.target.files[0]) {
      if (checkFileSize(e.target.files[0])) {
        renderFile(e.target.files[0]);
        setImageFile(e.target.files[0]);
      }
    }
  };
  const handleFileClick = () => {
    if (page == 2) return;
    fileInputRef.current.click();
  };

  return (
    <div className="flex flex-col justify-center items-center w-full">
      {images[page] ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragleave}
          onClick={handleFileClick}
          className="mt-3 bg-white p-4 border-header-blue border-4 border-solid rounded-2xl shadow-lg h-3/5 max-w-imageLoader"
        >
          <img src={images[page]} alt="image"></img>
        </div>
      ) : (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragleave}
          onClick={handleFileClick}
          className={`mt-3 p-4 border-header-blue border-4 border-solid rounded-2xl shadow-lg min-h-imageLoader content-center max-w-imageLoader justify-center ${
            dragOver ? "bg-blue border-darkblue" : "bg-white"
          }`}
        >
          <p className="font-Outfit text-3xl text-darkgray text-center">Posetive</p>
          <p className=" text-darkgray  text-center">클릭 및 끌어놓기로 사진 업로드</p>
        </div>
      )}
      {page === 2 ? (
        <ResultMenuBar />
      ) : (
        <div className="mt-10">
          <input
            ref={fileInputRef}
            className="file hidden"
            id="chooseFile"
            type="file"
            onChange={handleFile}
            accept="image/png, image/jpeg, image/gif"
            multiple={false}
          />
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
