package com.posetive.util;

import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.PutObjectRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.UUID;


@Component
public class S3Util {
    @Value("${application.bucket.name}")
    private String bucketName;

    @Autowired
    private AmazonS3Client amazonS3Client;

    @Transactional
    public String uploadFile(MultipartFile file){
        String fileName = createFileName(file.getOriginalFilename());

        ObjectMetadata metadata = new ObjectMetadata();
        metadata.setContentType(file.getContentType());
        metadata.setContentLength(file.getSize());

        try {
            PutObjectRequest request = new PutObjectRequest(bucketName, fileName, file.getInputStream(), metadata);
            amazonS3Client.putObject(request);      //upload
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return amazonS3Client.getUrl(bucketName, fileName).toString();
    }

    public String createFileName(String fileName){
        return UUID.randomUUID().toString().concat("_" + fileName);
    }
}
