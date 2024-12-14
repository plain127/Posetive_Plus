import React from "react";
import ReactPaginate from "react-paginate";

const Pagination = ({ pageCount, onPageChange, currentPage }) => {
  const pageRangeDisplayed = 5;
  const marginPagesDisplayed = 2;
  return (
    <ReactPaginate
      previousLabel="<"
      nextLabel=">"
      pageCount={pageCount}
      onPageChange={onPageChange}
      containerClassName={"pagination flex flex-row"}
      pageLinkClassName={"pagination__link"}
      activeLinkClassName={"pagination__link__active bg-yellow rounded-full "}
      currentPage={currentPage}
      pageRangeDisplayed={pageRangeDisplayed}
      marginPagesDisplayed={marginPagesDisplayed}
      className="p-2 flex flex-row min-w-24 justify-between"
    />
  );
};

export default Pagination;
