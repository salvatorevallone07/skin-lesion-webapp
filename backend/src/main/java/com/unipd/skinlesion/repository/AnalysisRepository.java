package com.unipd.skinlesion.repository;

import com.unipd.skinlesion.domain.Analysis;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.UUID;

public interface AnalysisRepository extends JpaRepository<Analysis, UUID> {
}
