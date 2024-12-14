import React, { useEffect, useState } from "react";
import GenerateImage from "./GenerateImage";
import { useNavigate } from "react-router-dom";
import { getCookie } from "../components/Cookies";

const GeneratePG = () => {
  const navigate = useNavigate();
  const loginId = localStorage.getItem("loginId");
  useEffect(() => {
    if (!getCookie("accessToken")) {
      navigate("/");
    } else if (loginId !== "admin09") {
      window.alert("권한이 없습니다. 관리자에게 문의하세요.");
      navigate("/");
      return;
    }
  }, []);
  return (
    <>
      {loginId !== "admin09" ? (
        <></>
      ) : (
        <div className="min-h-screen">
          <div className="bg-gradient-to-br from-blue-400 via-blue-300 to-blue-200 rounded-xl shadow-md mt-5 mx-5 p-4">
            basic version
          </div>
          <GenerateImage />
          <div className="mt-14"></div>
        </div>
      )}
    </>
  );
};

export default GeneratePG;
