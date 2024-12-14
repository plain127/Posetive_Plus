package com.posetive.api.controller;

import com.posetive.api.service.GenerationService;
import com.posetive.dto.response.ApiResponse;
import com.posetive.util.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequiredArgsConstructor
@RequestMapping("/generation")
public class GenerationController {
    private final GenerationService generationService;
    private final JwtUtil jwtUtil;

    @PostMapping("/pgpg")
    public ResponseEntity<ApiResponse> pgpgGeneration(@RequestParam("conditionImage") MultipartFile conditionImage, @RequestParam("targetImage") MultipartFile targetImage) throws IOException {
        Long userId = jwtUtil.getUserId();
        return ResponseEntity.ok().body(new ApiResponse<>(201, "PG2 이미지 생성 인퍼런스 성공", generationService.pgpgGeneration(conditionImage, targetImage, userId)));
    }

    @GetMapping("/my")
    public ResponseEntity<ApiResponse> getMyGenerationList() {
        Long userId = jwtUtil.getUserId();
        return ResponseEntity.ok().body(new ApiResponse<>(200, "내 이미지 생성 조회 완료", generationService.getMyGenerationList(userId)));
    }

    @GetMapping("/my/{generationId}")
    public ResponseEntity<ApiResponse> getMyGenerationDetail(@PathVariable Long generationId) {
        if (generationService.existsById(generationId)) {
            Long userId = jwtUtil.getUserId();
            if (generationService.isOwner(userId, generationId)) {
                return ResponseEntity.ok().body(new ApiResponse<>(200, "내 이미지 생성 개별 조회 완료", generationService.getMyGenerationDetail(generationId)));
            }
            return ResponseEntity.ok().body(new ApiResponse(401, "해당 이미지 생성 조회 권한이 없습니다", null));
        }
        return ResponseEntity.ok().body(new ApiResponse(404, "해당 이미지 생성이 존재하지 않거나 삭제되었습니다", null));
    }
}
