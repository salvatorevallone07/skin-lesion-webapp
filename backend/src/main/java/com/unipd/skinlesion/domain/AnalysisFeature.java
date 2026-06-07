package com.unipd.skinlesion.domain;

import jakarta.persistence.*;

@Entity
@Table(name = "analysis_feature")
public class AnalysisFeature {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "analysis_id")
    private Analysis analysis;

    @Column(name = "feature_name", nullable = false)
    private String featureName;

    @Column(name = "feature_value", nullable = false)
    private Double featureValue;

    @Column(name = "feature_index")
    private Integer featureIndex;

    public AnalysisFeature() { }

    public AnalysisFeature(String name, Double value, Integer index) {
        this.featureName = name;
        this.featureValue = value;
        this.featureIndex = index;
    }

    public Long getId() { return id; }
    public Analysis getAnalysis() { return analysis; }
    public void setAnalysis(Analysis analysis) { this.analysis = analysis; }
    public String getFeatureName() { return featureName; }
    public void setFeatureName(String v) { this.featureName = v; }
    public Double getFeatureValue() { return featureValue; }
    public void setFeatureValue(Double v) { this.featureValue = v; }
    public Integer getFeatureIndex() { return featureIndex; }
    public void setFeatureIndex(Integer v) { this.featureIndex = v; }
}
