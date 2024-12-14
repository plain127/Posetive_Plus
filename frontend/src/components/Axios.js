import axios from "axios";
import { getCookie } from "./Cookies";

const server = process.env.REACT_APP_SERVER_PORT;

// certification
export const axiosSignUp = async (userInfo) => {
  return axios({
    method: "post",
    url: `${server}/api/auth/signup`,
    data: userInfo,
  });
};

export const axiosLogin = async (userInfo) => {
  return axios({
    method: "post",
    url: `${server}/api/auth/login`,
    data: userInfo,
  });
};

export const axiosLogout = async () => {
  return axios({
    method: "post",
    url: `${server}/api/auth/logout`,
    headers: {
      Authorization: "Bearer " + getCookie("accessToken"),
    },
  });
};

export const axiosWithdraw = async () => {
  return axios({
    method: "put",
    url: `${server}/api/user/withdraw`,
    headers: {
      Authorization: "Bearer " + getCookie("accessToken"),
    },
  });
};

export const axiosResetPw = async (userInfo) => {
  return axios({
    method: "put",
    url: `${server}/api/auth/reset-pw`,
    headers: {
      Authorization: "Bearer " + getCookie("accessToken"),
    },
    data: userInfo,
  });
};

// Generate Image
export const axiosGenerationPgpg = async (imageInfo) => {
  let formData = new FormData();
  formData.append("conditionImage", imageInfo[0]);
  formData.append("targetImage", imageInfo[1]);
  return axios({
    method: "post",
    url: `${server}/api/generation/pgpg`,
    timeout:300000,
    headers: {
      Authorization: "Bearer " + getCookie("accessToken"),
      "Content-Type": "multipart/form-data",
    },
    data: formData,
  });
};

export const axiosMyPageList = async () => {
  return axios({
    method: "get",
    url: `${server}/api/generation/my`,
    headers: {
      Authorization: "Bearer " + getCookie("accessToken"),
    },
  });
};

export const axiosMyPageDetail = async (generation_id) => {
  return axios({
    method: "get",
    url: `${server}/api/generation/my/${generation_id}`,
    headers: {
      Authorization: "Bearer " + getCookie("accessToken"),
    },
  });
};
