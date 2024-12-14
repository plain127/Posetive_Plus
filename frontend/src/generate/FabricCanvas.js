import React, { useEffect, useRef, useState } from "react";
import { fabric } from "fabric";
import eraser from "../asset/eraser.png";
import pen from "../asset/pen.png";

const FabricCanvas = ({ setImageFile, setImage }) => {
  const canvasRef = useRef(null);
  const [canvas, setCanvas] = useState(null);
  const [saveDraw, setSaveDraw] = useState(false);
  useEffect(() => {
    const newCanvas = new fabric.Canvas(canvasRef.current, {
      height: 500,
      backgroundColor:"white"
    });
    setCanvas(newCanvas);
    newCanvas.freeDrawingBrush.width = 4;
    newCanvas.isDrawingMode = true;
    return () => {
      newCanvas.dispose();
    };
  }, []);

  const selectPen = () => {
    canvas.freeDrawingBrush.width = 4;
    canvas.freeDrawingBrush.color = "black";
    canvas.isDrawingMode = true;
  };
  const selectEraser = () => {
    canvas.freeDrawingBrush.color = "white";
    canvas.freeDrawingBrush.width = 20;
    canvas.isDrawingMode = true;
  };
  function b64toBlob(b64Data, contentType = "", sliceSize = 512) {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);

      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }

      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }

    const blob = new Blob(byteArrays, { type: contentType });
    return blob;
  }
  const saveImage = () => {
    const canvasImage = canvas.toDataURL("image/jpg");
    const base64Res = canvasImage.split(";base64,").pop();
    const blob = b64toBlob(base64Res, "image/jpeg");
    const canvasFile = new File([blob], "target.jpg", { type: "image/jpeg" });
    setImageFile(canvasFile);
    setImage(canvasImage);
    canvas.isDrawingMode = false;
    console.log(fabric.Path);
    canvas.selection = false;
    setSaveDraw(true);
  };

  return (
    <div
      className="flex flex-col items-center justify-center mt-3 mb-6"
    >
      <div className="container px-auto flex justify-center">
        <canvas
          id="fabric-canvas"
          className="max-w-imageLoader border-4 border-solid shadow-lg border-header-blue rounded-2xl"
          ref={canvasRef}
        ></canvas>
      </div>
      <div className="flex flex-row mt-5 w-80 items-center justify-around w-full max-w-imageLoader">
        <button className="w-8 h-8" onClick={selectPen}>
          <img src={pen} alt="pen"></img>
        </button>
        <button className="w-7 h-7" onClick={selectEraser}>
          <img src={eraser} alt="eraser"></img>
        </button>
        <div
          onClick={saveImage}
          className={`rounded-md text-white h-14 w-36 text-xl text-center content-center hover:cursor-pointer ${
            saveDraw ? "bg-green-200" : "bg-slate-400"
          }`}
        >
          그림 등록
        </div>
      </div>
    </div>
  );
};

export default FabricCanvas;
