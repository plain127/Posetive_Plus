package com.posetive.api.service;

import com.posetive.api.repository.UserRepository;
import com.posetive.dto.request.user.ResetNickNameReq;
import com.posetive.dto.request.user.ResetPasswordReq;
import com.posetive.entity.User;
import com.posetive.entity.UserStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public Boolean existsByLoginId(String loginId) { return userRepository.existsByLoginId(loginId); }

    public User findByLoginId(String loginId) { return userRepository.findByLoginId(loginId); }

    public void withdrawUser(Long userId) {
        User user = userRepository.findById(userId).orElse(null);
        user.setLoginId("deleted");
        user.setNickName("deleted");
        user.setUserStatus(UserStatus.WITHDRAWN);
    }

    public void resetPassword(ResetPasswordReq resetPasswordReq) {
        User user = userRepository.findByLoginId(resetPasswordReq.getLoginId());
        user.setPassword(passwordEncoder.encode(resetPasswordReq.getPassword()));
    }

    public Boolean existsByNickName(String nickName) { return userRepository.existsByNickName(nickName);}

    public void resetNickName(ResetNickNameReq resetNickNameReq, Long userId) {
        User user = userRepository.findById(userId).orElse(null);
        user.setNickName(resetNickNameReq.getNickName());
    }
}
