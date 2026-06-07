package com.unipd.skinlesion.service;

import com.unipd.skinlesion.domain.*;
import com.unipd.skinlesion.dto.AnalysisResponse;
import com.unipd.skinlesion.dto.PredictionResponse;
import com.unipd.skinlesion.repository.AnalysisRepository;
import com.unipd.skinlesion.repository.MlModelRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.time.OffsetDateTime;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class AnalysisService {

    private final InferenceClient inferenceClient;
    private final StorageService storageService;
    private final AnalysisRepository analysisRepository;
    private final MlModelRepository mlModelRepository;

    public AnalysisService(InferenceClient inferenceClient,
                           StorageService storageService,
                           AnalysisRepository analysisRepository,
                           MlModelRepository mlModelRepository) {
        this.inferenceClient = inferenceClient;
        this.storageService = storageService;
        this.analysisRepository = analysisRepository;
        this.mlModelRepository = mlModelRepository;
    }

    @Transactional
    public AnalysisResponse analyze(MultipartFile file) {
        UUID id = UUID.randomUUID();
        String path = storageService.store(id, file);

        Analysis analysis = new Analysis();
        analysis.setCreatedAt(OffsetDateTime.now());
        analysis.setOriginalFilename(file.getOriginalFilename());
        analysis.setContentType(file.getContentType());
        analysis.setFileSizeBytes(file.getSize());
        analysis.setImagePath(path);

        boolean modelTrained = false;
        long start = System.currentTimeMillis();
        try {
            PredictionResponse p = inferenceClient.predict(file);
            modelTrained = Boolean.TRUE.equals(p.modelTrained);

            MlModel model = resolveModel(p);
            analysis.setModel(model);
            analysis.setPredictedClass(p.predictedClass);
            analysis.setProbabilityMelanoma(p.probabilityMelanoma);
            analysis.setThresholdUsed(p.thresholdUsed);
            analysis.setImageWidth(p.imageWidth);
            analysis.setImageHeight(p.imageHeight);
            analysis.setInferenceStatus(InferenceStatus.SUCCESS);

            if (p.features != null) {
                AtomicInteger idx = new AtomicInteger(0);
                p.features.forEach((name, value) ->
                        analysis.addFeature(new AnalysisFeature(name, value, idx.getAndIncrement())));
            }
        } catch (Exception e) {
            analysis.setInferenceStatus(InferenceStatus.FAILED);
            analysis.setErrorMessage(e.getMessage());
        } finally {
            analysis.setInferenceMs((int) (System.currentTimeMillis() - start));
        }

        Analysis saved = analysisRepository.save(analysis);
        return AnalysisResponse.from(saved, modelTrained);
    }

    @Transactional(readOnly = true)
    public AnalysisResponse get(UUID id) {
        Analysis a = analysisRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Analysis not found: " + id));
        boolean trained = a.getModel() != null && !"untrained-dev".equals(a.getModel().getVersion());
        return AnalysisResponse.from(a, trained);
    }

    @Transactional(readOnly = true)
    public byte[] imageBytes(UUID id) {
        Analysis a = analysisRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Analysis not found: " + id));
        return storageService.read(a.getImagePath());
    }

    @Transactional(readOnly = true)
    public String imageContentType(UUID id) {
        return analysisRepository.findById(id).map(Analysis::getContentType).orElse("application/octet-stream");
    }

    private MlModel resolveModel(PredictionResponse p) {
        String version = p.modelVersion != null ? p.modelVersion : "unknown";
        return mlModelRepository.findByNameAndVersion("skin-lesion-classifier", version)
                .orElseGet(() -> {
                    MlModel m = new MlModel("skin-lesion-classifier", version);
                    m.setFramework("pytorch");
                    m.setInputDim(20);
                    m.setThreshold(p.thresholdUsed);
                    m.setTrainedAt(OffsetDateTime.now());
                    m.setNotes(Boolean.TRUE.equals(p.modelTrained)
                            ? "Trained on ISIC 2019 (MEL vs NV)"
                            : "Untrained dev model - predictions not meaningful");
                    return mlModelRepository.save(m);
                });
    }
}
