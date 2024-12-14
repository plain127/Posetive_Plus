package com.posetive.util;

import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

@Component
public class NickNameUtil {
    public String createNickName() throws IOException {
        URL url = new URL("https://www.rivestsoft.com/nickname/getRandomNickname.ajax");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");

        BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        String inputLine;
        StringBuilder response = new StringBuilder();

        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();

        String jsonString = response.toString();
        int dataIndex = jsonString.indexOf("\"data\":");
        int startIndex = jsonString.indexOf("\"", dataIndex + 7);
        int endIndex = jsonString.indexOf("\"", startIndex + 1);
        String name = jsonString.substring(startIndex + 1, endIndex);
        String key = " " + (int) (Math.random()*100000);

        return name + key;
    }
}
