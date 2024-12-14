package com.posetive.api.service;

import com.posetive.api.repository.UserRepository;
import com.posetive.dto.request.auth.RegisterUserReq;
import com.posetive.dto.response.auth.LoginInfoRes;
import com.posetive.entity.User;
import com.posetive.entity.UserStatus;
import com.posetive.util.CookieUtil;
import com.posetive.util.JwtUtil;
import com.posetive.util.NickNameUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Service
@Transactional
@RequiredArgsConstructor
public class AuthService {
    private final PasswordEncoder passwordEncoder;
    private final UserRepository userRepository;
    private final JwtUtil jwtUtil;
    private final CookieUtil cookieUtil;
    private final NickNameUtil nickNameUtil;

    public void registerUser(RegisterUserReq registerUserReq) throws IOException {
        User user = User.builder()
                .loginId(registerUserReq.getLoginId())
                .password(passwordEncoder.encode(registerUserReq.getPassword()))
                .registeredDate(LocalDateTime.now())
                .userStatus(UserStatus.ACTIVATED)
                .isSubscribed(false)
                .nickName(nickNameUtil.createNickName())
                .build();
        userRepository.save(user);
    }

    public String loginUser(String loginId, HttpServletResponse response) {
        User user = userRepository.findByLoginId(loginId);

        Map<String, String> userInfo = new HashMap<>();
        userInfo.put("id", user.getId() + "");

        String accessToken = jwtUtil.createAccessToken("user", userInfo, "user");
//        Cookie accessCookie = cookieUtil.addAccessCookie(accessToken);
//        response.addCookie(accessCookie);

        String refreshToken = jwtUtil.createRefreshToken("user", userInfo, "user");
        Cookie refreshCookie = cookieUtil.addRefreshCookie(refreshToken);
        response.addCookie(refreshCookie);
        return accessToken;
    }

    public LoginInfoRes getUserLoginInfo(Long userId, String accessToken, Boolean isSubscribed) {
        LoginInfoRes loginInfoRes = LoginInfoRes.builder()
                .accessToken(accessToken)
                .userId(userId)
                .nickName(userRepository.findById(userId).orElse(null).getNickName())
                .isSubscribed(isSubscribed)
                .build();

        return loginInfoRes;
    }

    public Boolean checkPassword(String password, String userPassword) {
        return passwordEncoder.matches(password, userPassword);
    }

    public void logoutUser(HttpServletRequest request, HttpServletResponse response, HttpSession session) {
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
//            Cookie accessCookie = new Cookie("accessToken", null);
//            accessCookie.setMaxAge(0);
//            accessCookie.setPath("/");
//            response.addCookie(accessCookie);

            Cookie refreshCookie = new Cookie("refreshToken", null);
            refreshCookie.setMaxAge(0);
            refreshCookie.setPath("/");
            response.addCookie(refreshCookie);
        }
        session.invalidate();
    }
}
