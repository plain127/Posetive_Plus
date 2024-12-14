import { Cookies } from "react-cookie";

const cookies = new Cookies();

export const setCookie = (name, value, option) => {
  return cookies.set(name, value, { ...option });
};

export const getCookie = (name) => {
  const cookie = cookies.get(name)
  if (cookie){}
  return cookie;
};

export const removeCookie = (name, option) => {
  return cookies.remove(name, { ...option });
};