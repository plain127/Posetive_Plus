package com.posetive.api.controller;

import com.posetive.api.service.UserService;
import com.posetive.dto.request.user.ResetNickNameReq;
import com.posetive.dto.request.user.ResetPasswordReq;
import com.posetive.dto.response.ApiResponse;
import com.posetive.util.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/user")
public class UserController {
    private final UserService userService;
    private final JwtUtil jwtUtil;

    @PutMapping("/withdraw")
    public ResponseEntity<ApiResponse> withdraw() {
        Long userId = jwtUtil.getUserId();
        userService.withdrawUser(userId);
        return ResponseEntity.ok().body(new ApiResponse<>(200, "회원 탈퇴 성공", null));
    }

    @PutMapping("/reset-pw")
    public ResponseEntity<ApiResponse> resetPassword(@RequestBody ResetPasswordReq resetPasswordReq) {
        userService.resetPassword(resetPasswordReq);
        return ResponseEntity.ok().body(new ApiResponse(206, "비밀 번호 변경 성공", null));
    }

    @PutMapping("/reset-nickname")
    public ResponseEntity<ApiResponse> resetNickName(@RequestBody ResetNickNameReq resetNickNameReq) {
        if (userService.existsByNickName(resetNickNameReq.getNickName())) {
            return ResponseEntity.ok().body(new ApiResponse(409, "중복된 닉네임입니다", null));
        }
        Long userId = jwtUtil.getUserId();
        userService.resetNickName(resetNickNameReq, userId);
        return ResponseEntity.ok().body(new ApiResponse(206, "닉네임 변경 성공", null));
    }
}
