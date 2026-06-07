package com.unipd.skinlesion.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import jakarta.annotation.PostConstruct;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.*;
import java.util.UUID;

/** Persists uploaded images on the shared volume; the DB only keeps the path. */
@Service
public class StorageService {

    private final Path root;

    public StorageService(@Value("${storage.upload-dir}") String uploadDir) {
        this.root = Paths.get(uploadDir);
    }

    @PostConstruct
    void init() {
        try {
            Files.createDirectories(root);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create upload dir " + root, e);
        }
    }

    public String store(UUID id, MultipartFile file) {
        String original = file.getOriginalFilename() == null ? "upload" : file.getOriginalFilename();
        String ext = "";
        int dot = original.lastIndexOf('.');
        if (dot >= 0) ext = original.substring(dot);
        Path target = root.resolve(id + ext);
        try {
            Files.copy(file.getInputStream(), target, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot store file", e);
        }
        return target.toString();
    }

    public byte[] read(String path) {
        try {
            return Files.readAllBytes(Paths.get(path));
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read file " + path, e);
        }
    }
}
