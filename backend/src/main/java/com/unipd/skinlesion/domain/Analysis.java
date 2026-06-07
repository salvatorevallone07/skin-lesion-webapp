package com.unipd.skinlesion.domain;

import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "analysis")
public class Analysis {

    @Id
    @GeneratedValue
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "model_id")
    private MlModel model;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt = OffsetDateTime.now();

    @Column(name = "original_filename")
    private String originalFilename;

    @Column(name = "content_type")
    private String contentType;

    @Column(name = "file_size_bytes")
    private Long fileSizeBytes;

    @Column(name = "image_width")
    private Integer imageWidth;

    @Column(name = "image_height")
    private Integer imageHeight;

    /** Path on the shared volume (the image bytes are NOT stored in the DB). */
    @Column(name = "image_path")
    private String imagePath;

    @Column(name = "predicted_class")
    private String predictedClass;

    @Column(name = "probability_melanoma")
    private Double probabilityMelanoma;

    @Column(name = "threshold_used")
    private Double thresholdUsed;

    @Enumerated(EnumType.STRING)
    @Column(name = "inference_status", nullable = false)
    private InferenceStatus inferenceStatus;

    @Column(name = "inference_ms")
    private Integer inferenceMs;

    @Column(name = "error_message", columnDefinition = "text")
    private String errorMessage;

    @OneToMany(mappedBy = "analysis", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<AnalysisFeature> features = new ArrayList<>();

    public void addFeature(AnalysisFeature f) {
        f.setAnalysis(this);
        this.features.add(f);
    }

    public UUID getId() { return id; }
    public MlModel getModel() { return model; }
    public void setModel(MlModel model) { this.model = model; }
    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }
    public String getOriginalFilename() { return originalFilename; }
    public void setOriginalFilename(String v) { this.originalFilename = v; }
    public String getContentType() { return contentType; }
    public void setContentType(String v) { this.contentType = v; }
    public Long getFileSizeBytes() { return fileSizeBytes; }
    public void setFileSizeBytes(Long v) { this.fileSizeBytes = v; }
    public Integer getImageWidth() { return imageWidth; }
    public void setImageWidth(Integer v) { this.imageWidth = v; }
    public Integer getImageHeight() { return imageHeight; }
    public void setImageHeight(Integer v) { this.imageHeight = v; }
    public String getImagePath() { return imagePath; }
    public void setImagePath(String v) { this.imagePath = v; }
    public String getPredictedClass() { return predictedClass; }
    public void setPredictedClass(String v) { this.predictedClass = v; }
    public Double getProbabilityMelanoma() { return probabilityMelanoma; }
    public void setProbabilityMelanoma(Double v) { this.probabilityMelanoma = v; }
    public Double getThresholdUsed() { return thresholdUsed; }
    public void setThresholdUsed(Double v) { this.thresholdUsed = v; }
    public InferenceStatus getInferenceStatus() { return inferenceStatus; }
    public void setInferenceStatus(InferenceStatus v) { this.inferenceStatus = v; }
    public Integer getInferenceMs() { return inferenceMs; }
    public void setInferenceMs(Integer v) { this.inferenceMs = v; }
    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String v) { this.errorMessage = v; }
    public List<AnalysisFeature> getFeatures() { return features; }
}
