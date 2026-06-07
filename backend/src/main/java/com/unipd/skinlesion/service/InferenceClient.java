package com.unipd.skinlesion.service;

import com.unipd.skinlesion.dto.PredictionResponse;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;

/** Talks to the Python FastAPI inference microservice. */
@Service
public class InferenceClient {

    private final RestClient client;

    public InferenceClient(RestClient mlRestClient) {
        this.client = mlRestClient;
    }

    public PredictionResponse predict(MultipartFile file) {
        ByteArrayResource resource;
        try {
            byte[] bytes = file.getBytes();
            resource = new ByteArrayResource(bytes) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename() != null ? file.getOriginalFilename() : "upload";
                }
            };
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);

        return client.post()
                .uri("/predict")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(PredictionResponse.class);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> info() {
        return client.get().uri("/info").retrieve().body(Map.class);
    }
}
