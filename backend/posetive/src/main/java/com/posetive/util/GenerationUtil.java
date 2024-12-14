package com.posetive.util;

import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

@Component
public class GenerationUtil {
    @Value("${inference-server-public-ip}")
    private String inferenceServerPublicIp;

    public String pgpgInference(Long generationId, String conditionImageUrl, String targetImageUrl) throws IOException {
        URL url = new URL(inferenceServerPublicIp +"/pgpg");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        String requestBody = "{"
                + "\"generationId\": "+generationId+","
                + "\"conditionImageUrl\": \""+conditionImageUrl+"\","
                + "\"targetImageUrl\": \""+targetImageUrl+"\""
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
        JSONObject jsonObject = new JSONObject(jsonString);
        String resultImageUrl = jsonObject.getString("result");
        return resultImageUrl;
    }
}
