package com.unipd.skinlesion.dto;

import com.unipd.skinlesion.domain.Analysis;
import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

/** Response sent to the Angular front end. */
public class AnalysisResponse {
    public UUID id;
    public OffsetDateTime createdAt;
    public String originalFilename;
    public String predictedClass;
    public Double probabilityMelanoma;
    public Double thresholdUsed;
    public String modelVersion;
    public Boolean modelTrained;
    public Integer imageWidth;
    public Integer imageHeight;
    public Integer inferenceMs;
    public String status;
    public String imageUrl;
    public Map<String, Double> features = new LinkedHashMap<>();

    public static AnalysisResponse from(Analysis a, Boolean modelTrained) {
        AnalysisResponse r = new AnalysisResponse();
        r.id = a.getId();
        r.createdAt = a.getCreatedAt();
        r.originalFilename = a.getOriginalFilename();
        r.predictedClass = a.getPredictedClass();
        r.probabilityMelanoma = a.getProbabilityMelanoma();
        r.thresholdUsed = a.getThresholdUsed();
        r.modelVersion = a.getModel() != null ? a.getModel().getVersion() : null;
        r.modelTrained = modelTrained;
        r.imageWidth = a.getImageWidth();
        r.imageHeight = a.getImageHeight();
        r.inferenceMs = a.getInferenceMs();
        r.status = a.getInferenceStatus() != null ? a.getInferenceStatus().name() : null;
        r.imageUrl = "/api/analyses/" + a.getId() + "/image";
        a.getFeatures().forEach(f -> r.features.put(f.getFeatureName(), f.getFeatureValue()));
        return r;
    }
}
