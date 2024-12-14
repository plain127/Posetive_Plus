package com.posetive.util;

import io.jsonwebtoken.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpServletRequest;
import java.io.UnsupportedEncodingException;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class JwtUtil {
    @Value("${salt}")
    private String SALT;

    public <T> String createAccessToken(String key, T data, String subject) {
        Date now = new Date();
        String jwt = Jwts.builder()
                .setHeaderParam("typ", "JWT")
                .setIssuedAt(now)
                .setSubject(subject)
                .setExpiration(new Date(System.currentTimeMillis() + 24 * 60 * 60 * 1000))
                .claim(key, data)
                .signWith(SignatureAlgorithm.HS512, this.generateKey())
                .compact();
        return jwt;
    }

    public <T> String createRefreshToken(String key, T data, String subject) {
        Date now = new Date();
        String jwt = Jwts.builder()
                .setHeaderParam("typ", "JWT")
                .setIssuedAt(now)
                .setSubject(subject)
                .setExpiration(new Date(System.currentTimeMillis() + 7 * 24 * 60 * 60 * 1000))
                .claim(key, data)
                .signWith(SignatureAlgorithm.HS512, this.generateKey())
                .compact();
        return jwt;
    }

    private byte[] generateKey() {
        byte[] key = null;
        try {
            key = SALT.getBytes("UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        return key;
    }

    public Boolean isUsable(String token) {
        try {
            Jws<Claims> claims = Jwts.parser()
                    .setSigningKey(this.generateKey())
                    .parseClaimsJws(token);
        } catch (ExpiredJwtException e) {
            // 유효기간 초과
            System.out.println(e.getMessage());
            throw e;
        } catch (UnsupportedJwtException e) {
            // 형식이 일치하지 않는 JWT
            System.out.println(e.getMessage());
            throw e;
        } catch (MalformedJwtException e) {
            // JWT가 올바르게 구성되지 않았을 경우
            System.out.println(e.getMessage());
            throw e;
        } catch (SignatureException e) {
            // 기존 서명을 확인하지 못한 경우
            System.out.println(e.getMessage());
            throw e;
        } catch (IllegalArgumentException e) {
            // claims가 비어있는 경우
            System.out.println(e.getMessage());
            throw e;
        } catch (Exception e) {
            throw e;
        }

        return true;
    }

    public Map<String, Object> get(String key) {
        HttpServletRequest request = ((ServletRequestAttributes) RequestContextHolder.currentRequestAttributes()).getRequest();
        String bearer = request.getHeader("Authorization");
        if (bearer == null) {
            return null;
        }

        String jwt = bearer.split(" ")[1];
        Jws<Claims> claims = null;
        try {
            claims = Jwts.parser()
                    .setSigningKey(SALT.getBytes("UTF-8"))
                    .parseClaimsJws(jwt);
        } catch (ExpiredJwtException e) {
            throw e;
        } catch (Exception e) {
            e.printStackTrace();
//            throw new UnauthorizedException(); 의도가 무엇일까?
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> value = (LinkedHashMap<String, Object>) claims.getBody().get(key);
        return value;
    }

    public Map<String, Object> getUserInfo(String jwt) {
        Jws<Claims> claims = null;
        try {
            claims = Jwts.parser()
                    .setSigningKey(SALT.getBytes("UTF-8"))
                    .parseClaimsJws(jwt);
        } catch (ExpiredJwtException e) {
            return (Map<String, Object>) e.getClaims().get("user");
        } catch (Exception e) {
            e.printStackTrace();
//            throw new UnauthorizedException();
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> value = (LinkedHashMap<String, Object>) claims.getBody().get("user");
        return value;
    }

    public Long getUserId() {
        if (this.get("user") == null) {
            return null;
        }
        return Long.parseLong((String) this.get("user").get("id"));
    }
}