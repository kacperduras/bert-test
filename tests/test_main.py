import unittest
from src.main import DocumentAnalyzer


class TestDocumentAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = DocumentAnalyzer()

        # Artykuły naukowe
        self.scientific_articles = [
            """
            Postępy w Obliczeniach Kwantowych: Korekcja Błędów

            Streszczenie:
            Niniejsze badanie przedstawia najnowsze osiągnięcia w technikach korekcji
            błędów kwantowych. Prezentujemy nowatorskie podejście osiągające 99.9%
            wierności w operacjach kubitowych przy użyciu zmodyfikowanej architektury
            kodów powierzchniowych.

            Metodologia:
            Eksperyment wykorzystywał procesor kwantowy z 53 kubitami nadprzewodzącymi.
            Korekcję błędów zaimplementowano przy użyciu własnego protokołu łączącego
            kody powierzchniowe z technikami dynamicznego rozprzęgania.

            Wyniki:
            Czasy koherencji przekroczyły 100 mikrosekund, przy wierności bramek
            konsekwentnie powyżej 99.9%.

            Słowa kluczowe: obliczenia kwantowe, korekcja błędów, kody powierzchniowe
            """
        ]

        # Artykuły prasowe
        self.press_articles = [
            """
            Przełom: Nowy Chip AI Zaprezentowany

            Wiodąca firma technologiczna ogłosiła dziś przełomowy procesor AI,
            który obiecuje 10-krotną poprawę wydajności w porównaniu z obecnymi
            rozwiązaniami. Chip, opracowywany przez trzy lata, wykorzystuje
            innowacyjną technologię stackowania 3D.
            """
        ]

        # Dokumentacja techniczna
        self.technical_docs = [
            """
            1. Przegląd Systemu
            - Wysokowydajna platforma
            - Skalowalność horyzontalna

            2. Komponenty
            - Moduł przetwarzania
            - System cache

            3. Bezpieczeństwo
            - Szyfrowanie end-to-end
            - Audyt dostępu
            """
        ]

        # Pytania weryfikacyjne
        self.questions = {
            'scientific': [
                "Jakie są główne wyniki badań?",
                "Jaką metodologię zastosowano?",
                "Jakie są kluczowe wnioski?"
            ],
            'press': [
                "Jakie jest główne wydarzenie?",
                "Kiedy to się wydarzyło?",
                "Jakie są implikacje?"
            ],
            'technical': [
                "Jakie są komponenty systemu?",
                "Jak działa zabezpieczenie?",
                "Jakie są wymagania?"
            ]
        }

    def test_initialization(self) -> None:
        """Test inicjalizacji komponentów"""
        self.assertIsNotNone(self.analyzer.tokenizer)
        self.assertIsNotNone(self.analyzer.model)

    def test_scientific_articles(self) -> None:
        """Test analizy artykułów naukowych"""
        for article in self.scientific_articles:
            result = self.analyzer.calculate_document_probability(
                article,
                self.questions['scientific']
            )
            self.assertGreaterEqual(result['average'], 0)
            self.assertLessEqual(result['average'], 1)
            self.assertIsInstance(result['per_question'], dict)

    def test_press_articles(self) -> None:
        """Test analizy artykułów prasowych"""
        for article in self.press_articles:
            result = self.analyzer.calculate_document_probability(
                article,
                self.questions['press']
            )
            self.assertGreaterEqual(result['average'], 0)
            self.assertLessEqual(result['average'], 1)

    def test_technical_documentation(self) -> None:
        """Test analizy dokumentacji technicznej"""
        for doc in self.technical_docs:
            fragments = self.analyzer.find_relevant_fragments(doc, threshold=0.3)
            self.assertTrue(len(fragments) > 0)
            # Sprawdzamy czy którykolwiek fragment zawiera numerację lub punktory
            found_structure = False
            for fragment, _, _ in fragments:
                if any(line.strip().startswith(('1.', '2.', '3.', '-'))
                       for line in fragment.split('\n')):
                    found_structure = True
                    break
            self.assertTrue(found_structure)

    def test_cross_domain_analysis(self) -> None:
        """Test analizy między domenami"""
        all_documents = (
                self.scientific_articles +
                self.press_articles +
                self.technical_docs
        )

        for doc in all_documents:
            for question_set in self.questions.values():
                result = self.analyzer.calculate_document_probability(doc, question_set)
                self.assertIsInstance(result, dict)
                self.assertIn('average', result)
                self.assertIn('per_question', result)

    def test_empty_data(self) -> None:
        """Test obsługi pustych danych wejściowych"""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_document_probability("", self.questions['scientific'])

        with self.assertRaises(ValueError):
            self.analyzer.calculate_document_probability(
                self.scientific_articles[0],
                []
            )

    def test_long_documents(self) -> None:
        """Test obsługi długich dokumentów"""
        long_doc = "test " * 1000
        fragments = self.analyzer.find_relevant_fragments(long_doc)
        for fragment, _, _ in fragments:
            tokens = self.analyzer.tokenizer.encode(fragment)
            self.assertLess(len(tokens), 512)

    def test_special_characters(self) -> None:
        """Test obsługi znaków specjalnych"""
        special_chars_doc = "!@#$%^&*() Testowy dokument ze znakami specjalnymi ąęśćżźńół"
        try:
            result = self.analyzer.calculate_document_probability(
                special_chars_doc,
                self.questions['technical']
            )
            self.assertIsInstance(result, dict)
            self.assertIn('average', result)
        except Exception as e:
            self.fail(f"Błąd przy znakach specjalnych: {e}")


if __name__ == '__main__':
    unittest.main()
