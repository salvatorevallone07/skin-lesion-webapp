import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AnalysisResponse {
  id: string;
  createdAt: string;
  originalFilename: string;
  predictedClass: string;
  probabilityMelanoma: number;
  thresholdUsed: number;
  modelVersion: string;
  modelTrained: boolean;
  imageWidth: number;
  imageHeight: number;
  inferenceMs: number;
  status: string;
  imageUrl: string;
  features: { [name: string]: number };
}

@Injectable({ providedIn: 'root' })
export class AnalysisService {
  constructor(private http: HttpClient) {}

  analyze(file: File): Observable<AnalysisResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<AnalysisResponse>('/api/analyses', form);
  }
}
