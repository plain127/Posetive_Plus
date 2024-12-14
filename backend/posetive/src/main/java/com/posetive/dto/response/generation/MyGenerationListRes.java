package com.posetive.dto.response.generation;

import com.posetive.dto.response.generation.MyGenerationListItem;
import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class MyGenerationListRes {
    private List<MyGenerationListItem> generations;
}
