package com.posetive.util;

import org.springframework.stereotype.Component;

import javax.servlet.http.Cookie;

@Component
public class CookieUtil {
    public Cookie addRefreshCookie(String refreshToken) {
        Cookie cookie = new Cookie("refreshToken", refreshToken);
        cookie.setMaxAge(86400 * 1000);
        cookie.setSecure(true);
        cookie.setHttpOnly(true);
        cookie.setPath("/");
        return cookie;
    }

    public Cookie addAccessCookie(String accessToken) {
        Cookie accessCookie = new Cookie("accessToken", accessToken);
        accessCookie.setMaxAge((int) System.currentTimeMillis() * 1800 * 1000);
        accessCookie.setSecure(true);
        accessCookie.setHttpOnly(true);
        accessCookie.setPath("/");
        return accessCookie;
    }
}