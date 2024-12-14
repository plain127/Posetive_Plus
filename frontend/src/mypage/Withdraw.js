import React from "react";
import Modal from "../components/Modal";
import { axiosWithdraw } from "../components/Axios";
import { useNavigate } from "react-router-dom";
import { removeCookie } from "../components/Cookies";
import { options } from "../components/setupCertified";

const Withdraw = ({ isOpen, isClose, setIsClose }) => {
  const navigate = useNavigate()
    const onClose = () => {
    setIsClose(true);
  };
  const onWithDraw = async () => {
    axiosWithdraw()
      .then((res) => {
        if(res.data.goojoCode === 200){
            removeCookie('accessToken', options);
            navigate("/");
            window.location.reload();
        }else{
            window.alert(res.data.message)
        }
      })
      .catch((err) => {
        console.error(err);
      });
  };
  return (
    <Modal isOpen={isOpen} isClose={isClose} setIsClose={setIsClose}>
      <div className="flex flex-col justify-center items-center">
        <h1 className="text-2xl mb-5">회원탈퇴</h1>
        <div className="text-center">
          탈퇴하면 지금까지 만들었던 이미지가 영구 삭제 됩니다.
        </div>
        <div className="flex flex-row mt-12">
          <button onClick={onWithDraw}>
            탈퇴
          </button>
          <button onClick={onClose} className="ml-5">
            돌아가기
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default Withdraw;
