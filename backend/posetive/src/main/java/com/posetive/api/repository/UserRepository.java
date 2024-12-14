package com.posetive.api.repository;

import com.posetive.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {
    Boolean existsByLoginId(String loginId);

    User findByLoginId(String loginId);

    Boolean existsByNickName(String nickName);

    User findByNickName(String nickName);
}
