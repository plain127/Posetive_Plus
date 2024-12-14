package com.posetive.dto.response.generation;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class MyGenerationDetailRes {
    private String conditionImageUrl;
    private String targetImageUrl;
    private String resultImageUrl;
    private String generationModel;
}
