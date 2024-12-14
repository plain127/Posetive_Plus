package com.posetive.api.controller;

import com.posetive.dto.response.ApiResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;

@RestController
@RequiredArgsConstructor
@RequestMapping("/test")
public class TestController {
    @Value("${inference-server-public-ip}")
    private String inferenceServerPublicIp;

    @GetMapping()
    public ResponseEntity<ApiResponse> test() throws IOException {
        URL url = new URL(inferenceServerPublicIp +"/hello");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        String url_1 = "https://posetive-bucket.s3.ap-northeast-2.amazonaws.com/7192243e-5564-4b97-bab7-30dc50daaf30_karina.jpg";
        String url_2 = "https://posetive-bucket.s3.ap-northeast-2.amazonaws.com/200ae60c-6dcc-4663-a109-815d3c0fd4c7_test.webp";

        String requestBody = "{"
                + "\"generationId\": 3,"
                + "\"conditionImageUrl\": \""+url_1+"\","
                + "\"targetImageUrl\": \""+url_2+"\""
                + "}";

        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = requestBody.getBytes("utf-8");
            os.write(input, 0, input.length);
        }

        BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        String inputLine;
        StringBuilder response = new StringBuilder();

        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();

        String jsonString = response.toString();

        return ResponseEntity.ok().body(new ApiResponse(200, "테스트 성공", jsonString));
    }
}
