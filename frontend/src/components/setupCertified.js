//axios headers
export const headers = {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
  };
  
  //set cookie options
  export const options = {
    path: "/",
    secure: true,
    sameSite: "Strict",
    HttpOnly: " HttpOnly ",
    maxAge: 7200
  };