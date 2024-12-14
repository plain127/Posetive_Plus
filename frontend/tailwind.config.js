/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js}"],
  theme: {
    extend: {
      fontFamily: {
        Outfit: ["Outfit", "sans-serif"],
        NotoSans: ["Noto Sans KR", "sans-serif"],
      },
      height: {
        tooltip: "448px",
      },
      width: {
        tooltip: "257px",
      },
      minHeight: {
        imageLoader: "400px",
      },
      maxWidth:{
        imageLoader:"400px",
      },
      minWidth:{
        imageLoader:"300px",
      },
      colors: {
        "header-blue": "#C5E5FC",
        darkblue: "#4B5A7E",
        "login-input": "#F5F5F6",
        darkblue2: "#435B76",
        darkblue3: "#80B0FF",
        greenblue: "#BEE7E0",
        yellow: "#FDEAB5",
        gray: "rgba(0,0,0,0.1)",
        darkgray: "#BEBEBE",
      },
      keyframes: {
        fadeInDown: {
          "0%": { transform: "translateY(-20px)", opacity: 0 },
          "50%": { transform: "translateY(0)", opacity: 100 },
        },
        fadeOutUp: {
          "0%": { transform: "translateY(0)", opacity: 100 },
          "50%": { transform: "translateY(-20px)", opacity: 0 },
        },
      },
      animation: {
        fadeInDown: 'fadeInDown 0.3s ease-out forwards',
        fadeOutUp: 'fadeOutUp 0.3s ease-in forwards'
      },
    },
    plugins: [],
  },
};
