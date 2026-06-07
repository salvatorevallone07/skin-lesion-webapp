package com.unipd.skinlesion.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

/** Mirrors the JSON returned by the Python /predict endpoint. */
public class PredictionResponse {

    @JsonProperty("predicted_class")
    public String predictedClass;

    @JsonProperty("probability_melanoma")
    public Double probabilityMelanoma;

    @JsonProperty("threshold_used")
    public Double thresholdUsed;

    @JsonProperty("model_version")
    public String modelVersion;

    @JsonProperty("model_trained")
    public Boolean modelTrained;

    @JsonProperty("image_width")
    public Integer imageWidth;

    @JsonProperty("image_height")
    public Integer imageHeight;

    public Map<String, Double> features;
}
