import React from "react";
import ballon from "../asset/ballon.png";
import karina from "../asset/karina.jpg";
import pictogram from "../asset/pictogram.jpg";
import attention from "../asset/attention.png";
import useStore from "../store";

const Tooltip = ({ page }) => {
  const activeTooltip = useStore((state) => state.activeTooltip);
  const setActiveTooltip = useStore((state) => state.setActiveTooltip);
  const explainations = [
    ["1. 전신 사진", "2. 한 명만 나온 사진", "3. 배경이 깔끔한 사진"],
    ["1. 전신 픽토그램", "2. 하나의 픽토그램"],
  ];
  const images = [karina, pictogram];
  const bg = {
    backgroundImage: `url(${ballon})`,
  };
  return (
    <>
      <div className="relative z-50">
        <img
          alt="tooltip"
          src={attention}
          className="w-8 h-8 hover:cursor-pointer"
          onClick={setActiveTooltip}
        ></img>
        {activeTooltip ? (
          <div
            className="absolute bg-no-repeat bg-cover w-tooltip h-tooltip flex flex-col justify-center items-center mt-2"
            style={bg}
          >
            <img
              src={images[page]}
              alt="example-image"
              className="w-44 max-h-80"
            ></img>
            <div className="mt-3 text-xl">이런 사진이 좋아요</div>
            <ol className="">
              {explainations[page].map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ol>
          </div>
        ) : (
          <></>
        )}
      </div>
    </>
  );
};

export default Tooltip;
