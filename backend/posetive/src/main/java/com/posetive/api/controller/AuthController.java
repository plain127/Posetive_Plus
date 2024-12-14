package com.posetive.api.controller;

import com.posetive.api.service.AuthService;
import com.posetive.api.service.UserService;
import com.posetive.dto.request.auth.LoginReq;
import com.posetive.dto.response.ApiResponse;
import com.posetive.dto.request.auth.RegisterUserReq;
import com.posetive.entity.User;
import com.posetive.util.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;

@RestController
@RequiredArgsConstructor
@RequestMapping("/auth")
public class AuthController {
    private final AuthService authService;
    private final UserService userService;
    private final JwtUtil jwtUtil;

    @PostMapping("/signup")
    public ResponseEntity<ApiResponse> registerUser(@RequestBody RegisterUserReq registerUserReq, HttpServletResponse response) throws IOException {
        String loginId = registerUserReq.getLoginId();
        if (userService.existsByLoginId(loginId)) {
            return ResponseEntity.ok().body(new ApiResponse(409, "이미 해당 정보로 가입한 유저가 있습니다", null));
        }

        authService.registerUser(registerUserReq);
        User user = userService.findByLoginId(loginId);
        String accessToken = authService.loginUser(loginId, response);
        return ResponseEntity.ok().body(new ApiResponse(201, "유저 생성 성공", authService.getUserLoginInfo(user.getId(), accessToken, user.getIsSubscribed())));
    }

    @PostMapping("/login")
    public ResponseEntity<ApiResponse> login(@RequestBody LoginReq loginReq, HttpServletResponse response) throws IOException {
        String loginId = loginReq.getLoginId();
        String password = loginReq.getPassword();
        if (userService.existsByLoginId(loginId)) {
            User user = userService.findByLoginId(loginId);

            if (authService.checkPassword(password, user.getPassword())) {
                String accessToken = authService.loginUser(loginId, response);
                return ResponseEntity.ok().body(new ApiResponse(200, "로그인 성공", authService.getUserLoginInfo(user.getId(), accessToken, user.getIsSubscribed())));
            }
        }
        return ResponseEntity.ok().body(new ApiResponse(400, "아이디 또는 비밀번호가 올바르지 않습니다", null));
    }

    @PostMapping("/logout")
    public ResponseEntity<ApiResponse> logout(HttpServletRequest request, HttpServletResponse response, HttpSession session) throws IOException {
        String bearer = request.getHeader("Authorization");
        if (bearer == null || "".equals(bearer)) {
            return ResponseEntity.status(401).body(null);
        }
        authService.logoutUser(request, response, session);
        return ResponseEntity.ok().body(new ApiResponse<>(200, "로그아웃 성공", null));
    }

    @PostMapping("/force-logout")
    public ResponseEntity<ApiResponse> forceLogout(HttpServletRequest request, HttpServletResponse response, HttpSession session) throws IOException {
        authService.logoutUser(request, response, session);
        return ResponseEntity.ok().body(new ApiResponse<>(200, "강제 로그아웃 성공", null));
    }
}
