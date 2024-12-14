import React from 'react';

function NotFound(props) {
    return (
        <div className='container mx-auto flex flex-col h-full'>
            <div className='font-NotoSans text-6xl text-nowrap mt-6'>
                <div>404</div>
                <div>Not</div>
                <div>Found</div>
            </div>
            <div className='flex justify-center mt-20 mb-20'>
                <a href='/' className='bg-darkblue text-greenblue w-44 h-14 text-xl text-center  content-center'>back to main</a>
            </div>
            <div className='font-NotoSans text-6xl text-right text-nowrap mb-6'>
                <div>The page</div>
                <div>you are</div>
                <div>looking for</div>
                <div>doesn't exist</div>
            </div>
        </div>
    );
}

export default NotFound;