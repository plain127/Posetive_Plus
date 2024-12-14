import React, { useEffect, useState } from "react";
import useStore from "../store";
import { useNavigate } from "react-router-dom";
import { getCookie } from "../components/Cookies";
import { axiosMyPageList } from "../components/Axios";
import ResultImage from "./ResultImage";
import Pagination from "./Pagination";
import Withdraw from "./Withdraw";

const MyPage = () => {
  const myPageList = useStore((state) => state.myPageList);
  const setMyPageList = useStore((state) => state.setMyPageList);
  const [pageCount, setPageCount] = useState(1);
  const [currentPage, setCurrentPage] = useState(0);
  const [isOpenWithdraw, setIsOpenWithdraw] = useState(false);
  const [isClose, setIsClose] = useState(true);
  const nickName = localStorage.getItem("nickName");
  const dataPerPage =
    window.innerWidth > 540 ? 12 : window.innerWidth > 367 ? 10 : 5;
  let currentImages = myPageList.slice(
    currentPage * dataPerPage,
    (currentPage + 1) * dataPerPage
  );
  const navigate = useNavigate();
  useEffect(() => {
    if (!getCookie("accessToken")) {
      navigate("/");
    } else {
      getMyPageList();
    }
  }, []);
  useEffect(() => {
    currentImages = myPageList.slice(
      currentPage * dataPerPage,
      (currentPage + 1) * dataPerPage
    );
  }, [currentPage]);

  const getMyPageList = async () => {
    axiosMyPageList()
      .then((res) => {
        const reverseList = res.data.data.generations.reverse();
        setMyPageList(reverseList);
        const newPageCount = Math.ceil(reverseList.length / dataPerPage);
        setPageCount(newPageCount ? newPageCount : 1);
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const onPageChange = (e) => {
    setCurrentPage(e.selected);
  };

  const openWithDraw = () => {
    setIsOpenWithdraw(true);
    setIsClose(false);
  };

  return (
    <div className="flex flex-col pt-12 sm:px-5 md:px-8 lg:px-10 xl:px-12 2xl:px-16  items-center mb-10 w-screen">
      <div className="bg-gradient-to-br from-blue-400 via-blue-300 to-blue-200 rounded-xl shadow-md m-5 p-4">
        <span className="text-white">{nickName}</span>
        <span>의 갤러리</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 justify-items-center w-full px-5">
        {currentImages.map((item) => {
          return <ResultImage key={item.generationId} item={item} />;
        })}
      </div>
      <Pagination
        pageCount={pageCount}
        onPageChange={onPageChange}
        currentPage={currentPage}
      />
      <div className="flex flex-row mt-4 items-end w-full">
        <div onClick={openWithDraw} className="ml-2 hover:cursor-pointer">
          회원탈퇴
        </div>
      </div>
      {isClose ? (
        <></>
      ) : (
        <Withdraw
          isOpen={isOpenWithdraw}
          isClose={isClose}
          setIsClose={setIsClose}
        />
      )}
    </div>
  );
};

export default MyPage;
