import os
import re
import math
from collections import defaultdict, Counter
import json
import argparse

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    return text.split()

def preprocess_text(text):
    return clean_text(text)

class BookCollection:
    def __init__(self):
        self.books = {}
        self.fragments = []
        self.fragment_index = []
        self.next_book_id = 0

    def add_book(self, filepath, book_name=None):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2 and content.strip():
            sentences = re.split(r'[.!?]+', content)
            paragraphs = [s.strip() for s in sentences if s.strip()]
        
        if not paragraphs:
            paragraphs = [content.strip()]
        
        fragments = []
        for para in paragraphs:
            if len(para) > 10:
                fragments.append(para)
        
        book_id = self.next_book_id
        self.next_book_id += 1
        self.books[book_id] = {
            'name': book_name if book_name else os.path.basename(filepath),
            'filepath': filepath,
            'fragments': fragments
        }
        
        for frag in fragments:
            self.fragments.append((book_id, frag))
            self.fragment_index.append((book_id, len(fragments)-1))
        
        return book_id

    def list_books(self):
        return [(bid, self.books[bid]['name']) for bid in self.books]

    def get_fragment_text(self, book_id, frag_idx):
        return self.books[book_id]['fragments'][frag_idx]

    def get_all_fragments(self):
        result = []
        for bid, book in self.books.items():
            for idx, frag in enumerate(book['fragments']):
                result.append((frag, bid, idx))
        return result

class SearchEngine:
    def __init__(self, collection):
        self.collection = collection
        self.fragments = collection.get_all_fragments()
        self.document_vectors = {}
        self.idf = {}
        self._build_index()

    def _build_index(self):
        doc_freq = defaultdict(int)
        total_docs = len(self.fragments)
        for frag, _, _ in self.fragments:
            words = set(tokenize(preprocess_text(frag)))
            for w in words:
                doc_freq[w] += 1

        self.idf = {}
        for w, df in doc_freq.items():
            self.idf[w] = math.log(total_docs / (df + 1)) + 1.0

        for idx, (frag, book_id, frag_idx) in enumerate(self.fragments):
            tf = Counter(tokenize(preprocess_text(frag)))
            vector = {}
            for w, count in tf.items():
                vector[w] = count * self.idf.get(w, 1.0)
            self.document_vectors[(book_id, frag_idx)] = vector

    def _vectorize_query(self, query):
        query_tokens = tokenize(preprocess_text(query))
        tf = Counter(query_tokens)
        vec = {}
        for w, count in tf.items():
            vec[w] = count * self.idf.get(w, 1.0)
        return vec

    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        dot = sum(v1 * v2 for v1, v2 in zip(vec1.values(), vec2.values()))
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query, top_k=5):
        q_vec = self._vectorize_query(query)
        scores = []
        for (book_id, frag_idx), doc_vec in self.document_vectors.items():
            score = self._cosine_similarity(q_vec, doc_vec)
            if score > 0:
                scores.append((score, book_id, frag_idx))
        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, book_id, frag_idx in scores[:top_k]:
            text = self.collection.get_fragment_text(book_id, frag_idx)
            results.append((score, book_id, frag_idx, text))
        return results

    def answer_question(self, question, top_k=3):
        results = self.search(question, top_k=top_k)
        if not results:
            return "Ответ не найден в загруженных книгах.", []
        
        quotes = []
        answer_parts = []
        for score, book_id, frag_idx, text in results:
            book_name = self.collection.books[book_id]['name']
            quote = f"📖 {book_name} (фрагмент {frag_idx+1}):\n{text}\n"
            quotes.append(quote)
            answer_parts.append(text)
        
        full_answer = "На основе найденных отрывков:\n\n" + "\n\n".join(answer_parts)
        return full_answer, quotes

class ConsoleUI:
    def __init__(self):
        self.collection = BookCollection()
        self.engine = None

    def run(self):
        self._print_header()
        while True:
            self._print_menu()
            choice = input("Выберите действие (1-6): ").strip()
            if choice == '1':
                self._load_books()
            elif choice == '2':
                self._search_fragments()
            elif choice == '3':
                self._ask_question()
            elif choice == '4':
                self._list_books()
            elif choice == '5':
                self._about()
            elif choice == '6':
                print("Выход...")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")

    def _print_header(self):
        print("\n" + "="*60)
        print("  Умный поиск по книгам (RAG-система)")
        print("  Отборочный этап «Искусственный интеллект», 2026")
        print("="*60)

    def _print_menu(self):
        print("\n" + "-"*40)
        print("МЕНЮ:")
        print("1. Загрузить книги (файлы .txt)")
        print("2. Поиск фрагментов по запросу")
        print("3. Задать вопрос")
        print("4. Показать список загруженных книг")
        print("5. О программе")
        print("6. Выход")
        print("-"*40)

    def _load_books(self):
        print("\nЗагрузка книг...")
        print("Укажите пути к текстовым файлам (.txt), разделяя их запятыми или пробелами.")
        print("Пример: book1.txt book2.txt или C:\\books\\war_and_peace.txt")
        files_input = input("Файлы: ").strip()
        if not files_input:
            print("Ничего не введено.")
            return
        files = re.split(r'[,\s]+', files_input)
        loaded = 0
        for f in files:
            f = f.strip()
            if not f:
                continue
            if not f.endswith('.txt'):
                print(f"Пропущено (не .txt): {f}")
                continue
            try:
                book_id = self.collection.add_book(f)
                print(f"✅ Загружена: {self.collection.books[book_id]['name']}")
                loaded += 1
            except Exception as e:
                print(f"❌ Ошибка при загрузке {f}: {e}")
        if loaded > 0:
            print("Построение поискового индекса...")
            self.engine = SearchEngine(self.collection)
            print("Индекс готов.")
        else:
            print("Не удалось загрузить ни одной книги.")

    def _search_fragments(self):
        if not self.engine:
            print("Сначала загрузите книги (пункт 1).")
            return
        query = input("\nВведите запрос для поиска фрагментов: ").strip()
        if not query:
            print("Запрос не может быть пустым.")
            return
        results = self.engine.search(query, top_k=5)
        if not results:
            print("Ничего не найдено.")
            return
        print(f"\nНайдено {len(results)} фрагментов:")
        for i, (score, book_id, frag_idx, text) in enumerate(results, 1):
            book_name = self.collection.books[book_id]['name']
            print(f"\n{i}. Книга: {book_name} (фрагмент {frag_idx+1}) [релевантность: {score:.3f}]")
            snippet = text[:300] + "..." if len(text) > 300 else text
            print(f"   {snippet}")

    def _ask_question(self):
        if not self.engine:
            print("Сначала загрузите книги (пункт 1).")
            return
        question = input("\nВведите вопрос: ").strip()
        if not question:
            print("Вопрос не может быть пустым.")
            return
        answer, quotes = self.engine.answer_question(question, top_k=3)
        print("\n" + "="*60)
        print("ОТВЕТ:")
        print(answer)
        if quotes:
            print("\nИспользованные цитаты:")
            for q in quotes:
                print(q)
        print("="*60)

    def _list_books(self):
        books = self.collection.list_books()
        if not books:
            print("Книги не загружены.")
            return
        print("\nЗагруженные книги:")
        for bid, name in books:
            frag_count = len(self.collection.books[bid]['fragments'])
            print(f"  {bid}. {name} (фрагментов: {frag_count})")

    def _about(self):
        print("""
Умный поиск по книгам
Версия 1.0

Функции:
- Загрузка книг в формате .txt
- Поиск фрагментов по ключевым словам (TF-IDF + косинусное сходство)
- Ответ на вопрос с указанием цитат

Для работы не требуется внешних библиотек (кроме стандартных).
Автор: Команда проекта
        """)

def main():
    ui = ConsoleUI()
    ui.run()

if __name__ == "__main__":
    main()
