import React from 'react';

function Footer(props) {
    return (
        <div className='flex flex-col border-t-2 border-black border-solid'>
            <div className="font-Outfit text-2xl ml-4 mt-9">9jodae</div>
            <a href='https://github.com/goojodae' className='ml-4 mt-4' target='_blank'>github</a>
            <div className='ml-4 mt-2'>members</div>
            <div className='flex flex-row ml-6 text-sm mt-1'>
                <span className='mr-2'>강병욱</span>
                <span className='mr-2'>김태인</span>
                <span className='mr-2'>석예은</span>
                <span className='mr-2'>윤지현</span>
                <span className='mr-2'>이진주</span>
                <span className='mr-2'>한상우</span>
                <span className='mr-2'>홍정현</span>
            </div>
            <div className='text-right m-4 text-sm'>Copyrightⓒ 2024, by 9jodae </div>
        </div>
    );
}

export default Footer;