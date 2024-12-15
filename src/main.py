from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import numpy as np
from typing import List, Tuple, Dict
import logging
from datetime import datetime
import sys
from pathlib import Path


class DocumentAnalyzer:
    def __init__(self) -> None:
        self.logger = self._setup_logger()
        self.logger.info("Inicjalizacja DocumentAnalyzer")

        model_name = "deepset/bert-base-cased-squad2"
        self.logger.info(f"Ładowanie modelu: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        self.logger.info("Model załadowany pomyślnie")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DocumentAnalyzer')
        logger.setLevel(logging.INFO)

        # Tworzenie katalogu logs jeśli nie istnieje
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Handler do pliku
        log_file = log_dir / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Handler do konsoli
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Format logów
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def calculate_document_probability(self, document: str, questions: List[str]) -> Dict[str, float]:
        if not document or not questions:
            raise ValueError("Dokument i pytania nie mogą być puste")

        self.logger.debug(f"Analiza dokumentu o długości {len(document)} znaków")
        results_per_question = {}

        for question in questions:
            self.logger.debug(f"Przetwarzanie pytania: {question}")
            inputs = self.tokenizer(
                question,
                document,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                start_probs = torch.softmax(outputs.start_logits, dim=1)
                end_probs = torch.softmax(outputs.end_logits, dim=1)
                max_prob = torch.max(start_probs).item() * torch.max(end_probs).item()
                results_per_question[question] = max_prob

        avg_prob = float(np.mean(list(results_per_question.values())))
        self.logger.debug(f"Średnie prawdopodobieństwo: {avg_prob:.4f}")

        return {
            'average': avg_prob,
            'per_question': results_per_question
        }

    def find_relevant_fragments(self, document: str, window_size: int = 100, threshold: float = 0.5) -> List[
        Tuple[str, float, Dict[str, float]]]:
        if not document:
            raise ValueError("Dokument nie może być pusty")

        self.logger.info("Rozpoczęcie analizy fragmentów")
        words = document.split()
        fragments = []

        relevance_questions = [
            "O czym jest ten fragment?",
            "Jakie są główne informacje w tym fragmencie?",
            "Co jest najważniejsze w tym fragmencie?"
        ]

        for i in range(0, len(words), window_size // 2):
            fragment = ' '.join(words[i:i + window_size])
            results = self.calculate_document_probability(fragment, relevance_questions)

            if results['average'] >= threshold:
                fragments.append((
                    fragment,
                    results['average'],
                    results['per_question']
                ))
                self.logger.debug(f"Znaleziono istotny fragment (score: {results['average']:.4f})")

        sorted_fragments = sorted(fragments, key=lambda x: x[1], reverse=True)
        self.logger.info(f"Znaleziono {len(sorted_fragments)} istotnych fragmentów (próg: {threshold})")

        return sorted_fragments

    def analyze_documents(self, doc1: str, doc2: str, questions: List[str]) -> Dict:
        self.logger.info("Rozpoczęcie analizy porównawczej dokumentów")

        self.logger.info("Analiza dokumentu technicznego")
        doc1_results = self.calculate_document_probability(doc1, questions)

        self.logger.info("Analiza dokumentu biznesowego")
        doc2_results = self.calculate_document_probability(doc2, questions)

        difference = abs(doc1_results['average'] - doc2_results['average'])
        self.logger.info(f"Różnica między dokumentami: {difference:.4f}")

        return {
            'document1': doc1_results,
            'document2': doc2_results,
            'difference': difference
        }


def main() -> None:
    analyzer = DocumentAnalyzer()

    # Dokument techniczny
    document1 = """
    Architektura Mikroserwisowa w Systemach AI

    Nowoczesne systemy AI wymagają skalowalnej i elastycznej infrastruktury.
    Architektura mikroserwisowa dostarcza optymalne rozwiązanie, gdzie każdy
    komponent AI działa jako niezależny serwis.

    Główne komponenty:

    1. Serwis Przetwarzania Danych
    - Walidacja danych wejściowych
    - Transformacja formatów
    - Zarządzanie kolejką zadań

    2. Serwis Modelu AI
    - Równoważenie obciążenia
    - Monitoring wydajności
    - Automatyczne skalowanie

    3. Serwis Wyników
    - Agregacja rezultatów
    - Formatowanie odpowiedzi
    - Cache wyników

    Implementacja wykorzystuje Kubernetes do orkiestracji i Prometheus
    do monitoringu. Całość jest zabezpieczona przez system OAuth2.
    """

    # Dokument biznesowy
    document2 = """
    Transformacja Cyfrowa w Sektorze Finansowym

    Sektor finansowy przechodzi fundamentalną transformację cyfrową.
    Banki i instytucje finansowe wdrażają zaawansowane rozwiązania AI
    do automatyzacji procesów i poprawy obsługi klienta.

    Kluczowe Obszary:

    1. Analiza Ryzyka
    - Automatyczna ocena zdolności kredytowej
    - Wykrywanie fraudów
    - Prognozowanie ryzyka rynkowego

    2. Obsługa Klienta
    - Chatboty 24/7
    - Personalizacja ofert
    - Automatyzacja procesów

    3. Optymalizacja Operacji
    - Redukcja kosztów o 25%
    - Przyspieszenie procesów o 40%
    - Poprawa dokładności o 60%

    Inwestycje w technologie AI zwracają się średnio w ciągu 18 miesięcy.
    """

    questions = [
        "Jakie są główne komponenty systemu?",
        "Jak zaimplementowana jest architektura?",
        "Jakie technologie są wykorzystywane?",
        "Jakie są korzyści biznesowe?",
        "Jak mierzone są efekty?",
        "Jaki jest zwrot z inwestycji?",
        "Jakie są główne wyzwania?",
        "Jak wygląda proces wdrożenia?",
        "Jakie są rezultaty?",
        "Jakie są konkretne przykłady zastosowań?",
        "Jakie są mierzalne efekty?",
        "Jak działa system w praktyce?"
    ]

    try:
        analyzer.logger.info("ROZPOCZĘCIE ANALIZY DOKUMENTÓW")

        results = analyzer.analyze_documents(document1, document2, questions)

        analyzer.logger.info("WYNIKI ANALIZY:")

        analyzer.logger.info("Dokument techniczny:")
        analyzer.logger.info(f"Ogólny wynik: {results['document1']['average']:.2%}")

        for q, score in sorted(results['document1']['per_question'].items(), key=lambda x: x[1], reverse=True)[:3]:
            analyzer.logger.info(f"Pytanie: {q} - Wynik: {score:.2%}")

        analyzer.logger.info("Dokument biznesowy:")
        analyzer.logger.info(f"Ogólny wynik: {results['document2']['average']:.2%}")

        for q, score in sorted(results['document2']['per_question'].items(), key=lambda x: x[1], reverse=True)[:3]:
            analyzer.logger.info(f"Pytanie: {q} - Wynik: {score:.2%}")

        analyzer.logger.info(f"Różnica między dokumentami: {results['difference']:.2%}")

        better_doc = document1 if results['document1']['average'] > results['document2']['average'] else document2
        doc_type = "techniczny" if results['document1']['average'] > results['document2']['average'] else "biznesowy"

        analyzer.logger.info(f"SZCZEGÓŁOWA ANALIZA DOKUMENTU")
        fragments = analyzer.find_relevant_fragments(better_doc, threshold=0.6)

        for i, (fragment, score, questions_scores) in enumerate(fragments[:3], 1):
            analyzer.logger.info(f"Fragment {i}:")
            analyzer.logger.info(f"Tekst: {fragment.strip()}")
            analyzer.logger.info(f"Ocena całkowita: {score:.2%}")

            for q, s in questions_scores.items():
                analyzer.logger.info(f"Pytanie: {q} - Wynik: {s:.2%}")

        analyzer.logger.info("PODSUMOWANIE:")
        analyzer.logger.info(f"• Dokument {doc_type} został oceniony jako bardziej odpowiedni")
        analyzer.logger.info(f"• Różnica w jakości dokumentów: {results['difference']:.2%}")
        analyzer.logger.info(f"• Znaleziono {len(fragments)} istotnych fragmentów")
        analyzer.logger.info(f"• Najwyższy wynik fragmentu: {max(score for _, score, _ in fragments):.2%}")

        analyzer.logger.info("Analiza zakończona pomyślnie")

    except Exception as e:
        analyzer.logger.error(f"Wystąpił błąd podczas analizy: {e}")


if __name__ == "__main__":
    main()
