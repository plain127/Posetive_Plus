package com.posetive.dto.response.auth;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class LoginInfoRes {
    private String accessToken;
    private Long userId;
    private String nickName;
    private Boolean isSubscribed;
}
