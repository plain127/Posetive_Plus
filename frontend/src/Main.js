import React from "react";
import logo from "../src/asset/logo-color.png";
import { getCookie } from "./components/Cookies";
import crown from "./asset/crown.png";

const Main = () => {
  return (
    <div className="container mx-auto px-4 flex flex-col pt-24 items-center h-screen w-full">
      <img src={logo} className=" w-72" alt="posetive-icon"></img>
      <p className="mb-10">이미지를 원하는 자세로 바꿔보세요</p>
      {getCookie("accessToken") ? (
        <div className="flex flex-col justify-around h-40 ">
          <a
            href="/generate/pgpg"
            className="bg-header-blue h-14 w-60 text-darkblue rounded-xl text-center content-center hover:shadow-xl transition duration-5000 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2"
          >
            basic model
          </a>
          <div className="relative h-14 w-60 text-white rounded-xl text-center content-center bg-gradient-to-r from-pink-300 to-purple-400 hover:from-purple-400 transition duration-7000 hover:to-pink-300 hover:shadow-xl  ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500">
            <a
              href="/generate/pidm"
              className="font-bold"
            >
              advanced pro model
            </a>
            <img src={crown} className="absolute left-32 top-2 w-5" alt="crown-icon"/>
          </div>
        </div>
      ) : (
        <div className="flex flex-col justify-around h-40">
          <a
            href="/login"
            className="bg-header-blue h-14 w-60 text-xl rounded-xl text-center content-center hover:shadow-xl transition duration-7000 ease-in-out"
          >
            로그인
          </a>
          <a
            href="/signup"
            className="bg-header-blue h-14 w-60 text-xl rounded-xl text-center content-center hover:shadow-xl transition duration-7000 ease-in-out"
          >
            회원가입
          </a>
        </div>
      )}
    </div>
  );
};

export default Main;
