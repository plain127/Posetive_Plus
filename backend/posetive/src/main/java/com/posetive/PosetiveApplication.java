package com.posetive;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import javax.annotation.PostConstruct;
import java.util.TimeZone;

@SpringBootApplication(
        exclude = {
            org.springframework.cloud.aws.autoconfigure.context.ContextInstanceDataAutoConfiguration.class,
            org.springframework.cloud.aws.autoconfigure.context.ContextStackAutoConfiguration.class,
            org.springframework.cloud.aws.autoconfigure.context.ContextRegionProviderAutoConfiguration.class
        }
)
public class PosetiveApplication {
    @PostConstruct
    public void started() {
        TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
    }

    public static void main(String[] args) {
        SpringApplication.run(PosetiveApplication.class, args);
    }
}