package com.unipd.skinlesion.domain;

import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "ml_model",
       uniqueConstraints = @UniqueConstraint(columnNames = {"name", "version"}))
public class MlModel {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private String version;

    private String framework;

    @Column(name = "input_dim")
    private Integer inputDim;

    private Double threshold;

    @Column(name = "trained_at")
    private OffsetDateTime trainedAt;

    private Double accuracy;

    @Column(columnDefinition = "text")
    private String notes;

    public MlModel() { }

    public MlModel(String name, String version) {
        this.name = name;
        this.version = version;
    }

    public Long getId() { return id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public String getFramework() { return framework; }
    public void setFramework(String framework) { this.framework = framework; }
    public Integer getInputDim() { return inputDim; }
    public void setInputDim(Integer inputDim) { this.inputDim = inputDim; }
    public Double getThreshold() { return threshold; }
    public void setThreshold(Double threshold) { this.threshold = threshold; }
    public OffsetDateTime getTrainedAt() { return trainedAt; }
    public void setTrainedAt(OffsetDateTime trainedAt) { this.trainedAt = trainedAt; }
    public Double getAccuracy() { return accuracy; }
    public void setAccuracy(Double accuracy) { this.accuracy = accuracy; }
    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
}
