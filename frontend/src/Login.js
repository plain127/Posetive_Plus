import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import Schema from "./components/Schema";
import { axiosLogin } from "./components/Axios";
import { getCookie, setCookie } from "./components/Cookies";
import { options } from "./components/setupCertified";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const {
    register,
    handleSubmit,
    formState: { isSubmitting, errors },
  } = useForm({ mode: "onChange", resolver: yupResolver(Schema) });
  const navigate = useNavigate();

  useEffect(() => {
    if (getCookie("accessToken")) navigate("/");
  }, []);

  const onSubmit = async (data) => {
    axiosLogin(data)
      .then((res) => {
        if(res.data.goojoCode === 200){
          window.alert("로그인되었습니다.");
          localStorage.setItem('loginId', data.loginId)
          localStorage.setItem('nickName',res.data.data.nickName)
          setCookie("accessToken", `${res.data.data.accessToken}`, options);
          navigate("/");
        window.location.reload();
        }else{
          window.alert(res.data.message)
        }
      })
      .catch((err) => {
        window.alert("아이디 혹은 비밀번호가 일치하지 않습니다.");
        console.error(err);
      });
  };

  return (
    <div className="container mx-auto px-4 flex flex-col pt-24 items-center h-screen w-full">
      <h1 className="font-Outfit text-3xl font-medium">Login</h1>
      <form
        onSubmit={handleSubmit(onSubmit)}
        className="flex flex-col m-6  items-center"
      >
        <input
          type="id"
          name="loginId"
          className="bg-login-input border-solid border-b-2 border-darkblue w-60 h-12 p-3"
          placeholder="아이디"
          {...register("loginId")}
        ></input>
        {errors.loginId ? <p className="">{errors.loginId.message}</p> : <></>}
        <input
          type="password"
          name="password"
          className="bg-login-input border-soild border-b-2 border-darkblue w-60 h-12 mt-3 p-3"
          placeholder="비밀번호"
          {...register("password")}
        ></input>
        {errors.password ? <p>{errors.password.message}</p> : <></>}
        <input
          type="submit"
          name="login"
          value="로그인"
          className="bg-darkblue2 h-12 w-60 text-greenblue mt-10"
          disabled={isSubmitting}
        ></input>
      </form>

      <div className="flex flex-row justify-around mt-24 w-60">
        <a href="/signup" className="text-darkblue text-sm">
          회원가입
        </a>
      </div>
    </div>
  );
};

export default Login;
