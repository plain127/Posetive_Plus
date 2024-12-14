import React from 'react';

const ResultMenuBar = () => {
    const resetPage = () => {
        window.location.reload()
    }
    return (
        <div className='flex justify-center mt-10'>
            <div className="bg-darkgray text-white h-14 w-60 text-xl text-center content-center hover:cursor-pointer" onClick={resetPage}>다시하기</div>
        </div>
    );
};

export default ResultMenuBar;