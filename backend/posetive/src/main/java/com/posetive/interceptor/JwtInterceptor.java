package com.posetive.interceptor;

import com.posetive.util.JwtUtil;
import io.jsonwebtoken.ExpiredJwtException;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpMethod;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@Component
@RequiredArgsConstructor
public class JwtInterceptor implements HandlerInterceptor {
    private static final String HEADER_AUTH = "Authorization";
    private final JwtUtil jwtUtil;

    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {

        // axios의 Preflight OPTION 요청 거름
        if (HttpMethod.OPTIONS.matches(request.getMethod())) {
            return true;
        }

        String bearer = request.getHeader(HEADER_AUTH);

        if (bearer != null && !"".equals(bearer)) {
            final String token = request.getHeader(HEADER_AUTH).split(" ")[1];
            try {
                if (token != null && !"".equals(bearer) && jwtUtil.isUsable(token)) {
                    return true;
                }
            } catch (ExpiredJwtException e) {
                response.sendError(401, "토큰이 만료되었습니다.");
            } catch (Exception e) {
                response.sendError(401, "토큰이 유효하지 않습니다.");
                System.out.println(e.getMessage());
            }
        }

//        Cookie accessCookie = new Cookie("accessToken", null);
//        accessCookie.setMaxAge(0);
//        accessCookie.setPath("/");
//        response.addCookie(accessCookie);

        Cookie refreshCookie = new Cookie("refreshToken", null);
        refreshCookie.setMaxAge(0);
        refreshCookie.setPath("/");
        response.addCookie(refreshCookie);

        response.sendError(401, "토큰이 비어있습니다");
        return false;
    }
}
