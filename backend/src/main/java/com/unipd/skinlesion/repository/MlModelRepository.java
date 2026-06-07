package com.unipd.skinlesion.repository;

import com.unipd.skinlesion.domain.MlModel;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface MlModelRepository extends JpaRepository<MlModel, Long> {
    Optional<MlModel> findByNameAndVersion(String name, String version);
}
