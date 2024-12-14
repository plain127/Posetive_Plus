import React, { useEffect, useState } from "react";

const Modal = ({ isOpen, isClose, setIsClose, children }) => {
  useEffect(() => {
    if (!isOpen) return;
  }, [isOpen]);

  useEffect(() => {
    if (isClose) return;
  }, [isClose]);

  const onClose = () => {
    setIsClose(true);
  };

  return (
    <div className="fixed inset-0 bg-gray flex justify-center items-center z-50">
      <div className="p-4 rounded-lg shadow-lg max-w-52 w-full h-80 bg-white">
        <div className="flex justify-end">
          <button onClick={onClose} className="text-xl font-bold p-2">
            &times;
          </button>
        </div>
        {children}
      </div>
    </div>
  );
};

export default Modal;
