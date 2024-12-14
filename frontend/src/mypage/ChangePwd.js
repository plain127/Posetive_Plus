import React from "react";
import Modal from "../components/Modal";

const ChangePwd = ({ isOpen, isClose, setIsClose }) => {
  return (
    <Modal isOpen={isOpen} isClose={isClose} setIsClose={setIsClose}>
      <div className="flex flex-row">
        <h1>비밀번호 변경</h1>
        <input></input>
      </div>
    </Modal>
  );
};

export default ChangePwd;
