import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AnalysisService, AnalysisResponse } from './analysis.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  selectedFile = signal<File | null>(null);
  previewUrl = signal<string | null>(null);
  loading = signal(false);
  error = signal<string | null>(null);
  result = signal<AnalysisResponse | null>(null);

  constructor(private service: AnalysisService) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files && input.files.length ? input.files[0] : null;
    this.result.set(null);
    this.error.set(null);
    this.selectedFile.set(file);
    if (this.previewUrl()) {
      URL.revokeObjectURL(this.previewUrl()!);
    }
    this.previewUrl.set(file ? URL.createObjectURL(file) : null);
  }

  analyze(): void {
    const file = this.selectedFile();
    if (!file) return;
    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);
    this.service.analyze(file).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.error || 'Errore durante l\'inferenza.');
        this.loading.set(false);
      },
    });
  }

  isMelanoma(): boolean {
    return this.result()?.predictedClass === 'melanoma';
  }

  classLabel(): string {
    const c = this.result()?.predictedClass;
    if (c === 'melanoma') return 'Melanoma';
    if (c === 'melanocytic_nevus') return 'Nevo melanocitico';
    return c ?? '';
  }

  probabilityPct(): string {
    const p = this.result()?.probabilityMelanoma;
    return p != null ? (p * 100).toFixed(1) + '%' : '-';
  }

  featureList(): { name: string; value: number }[] {
    const f = this.result()?.features;
    if (!f) return [];
    return Object.keys(f).map((name) => ({ name, value: f[name] }));
  }
}
