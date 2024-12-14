import React, { useEffect } from "react";
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import Schema from "./components/Schema";
import { axiosSignUp } from "./components/Axios";
import { getCookie, setCookie } from "./components/Cookies";
import { useNavigate } from "react-router-dom";
import { options } from "./components/setupCertified";

const SignUp = () => {
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
    axiosSignUp(data)
      .then((res) => {
        if (res.data.goojoCode === 201) {
          window.alert("회원가입되었습니다.");
          setCookie("accessToken", `${res.data.data.accessToken}`, options);
          window.location.reload();
        } else{
          window.alert(res.data.message)
          console.error(res)
        }
      })
      .catch((err) => {
        window.alert("회원가입 실패");
        console.error(err);
      });
  };

  return (
    <div className="container mx-auto px-4 flex flex-col pt-24 items-center h-screen w-full">
      <h1 className="font-Outfit text-3xl font-medium">Sign Up</h1>
      <form
        onSubmit={handleSubmit(onSubmit)}
        className="flex flex-col m-6 items-center"
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
        {errors.password ? (
          <p className="">{errors.password.message}</p>
        ) : (
          <></>
        )}
        <input
          type="password"
          name="passwordConfirm"
          className="bg-login-input border-soild border-b-2 border-darkblue w-60 h-12 mt-3 p-3"
          placeholder="비밀번호 확인"
          {...register("passwordConfirm")}
        ></input>
        {errors.passwordConfirm ? (
          <p className="">{errors.passwordConfirm.message}</p>
        ) : (
          <></>
        )}
        <input
          type="submit"
          name="signup"
          value="회원가입"
          className="bg-darkblue2 h-12 w-60 text-greenblue mt-10"
          disabled={isSubmitting}
        ></input>
      </form>

      <div className="flex flex-row justify-around mt-24 w-60">
        <a href="/login" className="text-darkblue text-sm">
          로그인
        </a>
      </div>
    </div>
  );
};

export default SignUp;
