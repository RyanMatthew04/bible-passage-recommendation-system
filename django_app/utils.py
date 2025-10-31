import re
import pandas as pd
import numpy as np
import faiss
from django.conf import settings
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


# Bible book name mappings
BOOK_MAPPINGS = {
    # Old Testament
    'genesis': 'genesis', 'gen': 'genesis', 'ge': 'genesis', 'gn': 'genesis',
    'exodus': 'exodus', 'exod': 'exodus', 'ex': 'exodus', 'exo': 'exodus',
    'leviticus': 'leviticus', 'lev': 'leviticus', 'le': 'leviticus', 'lv': 'leviticus',
    'numbers': 'numbers', 'num': 'numbers', 'nu': 'numbers', 'nm': 'numbers', 'nb': 'numbers',
    'deuteronomy': 'deuteronomy', 'deut': 'deuteronomy', 'dt': 'deuteronomy', 'de': 'deuteronomy',
    'joshua': 'joshua', 'josh': 'joshua', 'jos': 'joshua', 'jsh': 'joshua',
    'judges': 'judges', 'judg': 'judges', 'jdg': 'judges', 'jg': 'judges', 'jdgs': 'judges',
    'ruth': 'ruth', 'rth': 'ruth', 'ru': 'ruth',
    '1 samuel': '1samuel', '1samuel': '1samuel', '1sam': '1samuel', '1sa': '1samuel', '1s': '1samuel',
    'first samuel': '1samuel', 'i samuel': '1samuel',
    '2 samuel': '2samuel', '2samuel': '2samuel', '2sam': '2samuel', '2sa': '2samuel', '2s': '2samuel',
    'second samuel': '2samuel', 'ii samuel': '2samuel',
    '1 kings': '1kings', '1kings': '1kings', '1kgs': '1kings', '1ki': '1kings', '1k': '1kings',
    'first kings': '1kings', 'i kings': '1kings',
    '2 kings': '2kings', '2kings': '2kings', '2kgs': '2kings', '2ki': '2kings', '2k': '2kings',
    'second kings': '2kings', 'ii kings': '2kings',
    '1 chronicles': '1chronicles', '1chronicles': '1chronicles', '1chron': '1chronicles', '1chr': '1chronicles', '1ch': '1chronicles',
    'first chronicles': '1chronicles', 'i chronicles': '1chronicles',
    '2 chronicles': '2chronicles', '2chronicles': '2chronicles', '2chron': '2chronicles', '2chr': '2chronicles', '2ch': '2chronicles',
    'second chronicles': '2chronicles', 'ii chronicles': '2chronicles',
    'ezra': 'ezra', 'ezr': 'ezra',
    'nehemiah': 'nehemiah', 'neh': 'nehemiah', 'ne': 'nehemiah',
    'esther': 'esther', 'esth': 'esther', 'es': 'esther',
    'job': 'job', 'jb': 'job',
    'psalm': 'psalm', 'psalms': 'psalm', 'ps': 'psalm', 'pslm': 'psalm', 'psa': 'psalm', 'psm': 'psalm', 'pss': 'psalm',
    'proverbs': 'proverbs', 'prov': 'proverbs', 'pro': 'proverbs', 'prv': 'proverbs', 'pr': 'proverbs',
    'ecclesiastes': 'ecclesiastes', 'eccles': 'ecclesiastes', 'eccl': 'ecclesiastes', 'ec': 'ecclesiastes', 'ecc': 'ecclesiastes', 'qoh': 'ecclesiastes',
    'song of solomon': 'songofsolomon', 'song': 'songofsolomon', 'song of songs': 'songofsolomon', 'sos': 'songofsolomon', 'so': 'songofsolomon', 'songofsolomon': 'songofsolomon',
    'canticle of canticles': 'songofsolomon', 'canticles': 'songofsolomon', 'cant': 'songofsolomon',
    'isaiah': 'isaiah', 'isa': 'isaiah', 'is': 'isaiah',
    'jeremiah': 'jeremiah', 'jer': 'jeremiah', 'je': 'jeremiah', 'jr': 'jeremiah',
    'lamentations': 'lamentations', 'lam': 'lamentations', 'la': 'lamentations',
    'ezekiel': 'ezekiel', 'ezek': 'ezekiel', 'eze': 'ezekiel', 'ezk': 'ezekiel',
    'daniel': 'daniel', 'dan': 'daniel', 'da': 'daniel', 'dn': 'daniel',
    'hosea': 'hosea', 'hos': 'hosea', 'ho': 'hosea',
    'joel': 'joel', 'joe': 'joel', 'jl': 'joel',
    'amos': 'amos', 'amo': 'amos', 'am': 'amos',
    'obadiah': 'obadiah', 'obad': 'obadiah', 'ob': 'obadiah',
    'jonah': 'jonah', 'jnh': 'jonah', 'jon': 'jonah',
    'micah': 'micah', 'mic': 'micah', 'mc': 'micah',
    'nahum': 'nahum', 'nah': 'nahum', 'na': 'nahum',
    'habakkuk': 'habakkuk', 'hab': 'habakkuk', 'hb': 'habakkuk',
    'zephaniah': 'zephaniah', 'zeph': 'zephaniah', 'zep': 'zephaniah', 'zp': 'zephaniah',
    'haggai': 'haggai', 'hag': 'haggai', 'hg': 'haggai',
    'zechariah': 'zechariah', 'zech': 'zechariah', 'zec': 'zechariah', 'zc': 'zechariah',
    'malachi': 'malachi', 'mal': 'malachi', 'ml': 'malachi',
    
    # New Testament
    'matthew': 'matthew', 'matt': 'matthew', 'mat': 'matthew', 'mt': 'matthew',
    'mark': 'mark', 'mar': 'mark', 'mrk': 'mark', 'mk': 'mark', 'mr': 'mark',
    'luke': 'luke', 'luk': 'luke', 'lk': 'luke',
    'john': 'john', 'joh': 'john', 'jhn': 'john', 'jn': 'john',
    'acts': 'acts', 'act': 'acts', 'ac': 'acts',
    'romans': 'romans', 'rom': 'romans', 'ro': 'romans', 'rm': 'romans',
    '1 corinthians': '1corinthians', '1corinthians': '1corinthians', '1cor': '1corinthians', '1co': '1corinthians', '1c': '1corinthians',
    'first corinthians': '1corinthians', 'i corinthians': '1corinthians',
    '2 corinthians': '2corinthians', '2corinthians': '2corinthians', '2cor': '2corinthians', '2co': '2corinthians', '2c': '2corinthians',
    'second corinthians': '2corinthians', 'ii corinthians': '2corinthians',
    'galatians': 'galatians', 'gal': 'galatians', 'ga': 'galatians',
    'ephesians': 'ephesians', 'ephes': 'ephesians', 'eph': 'ephesians',
    'philippians': 'philippians', 'phil': 'philippians', 'php': 'philippians', 'pp': 'philippians',
    'colossians': 'colossians', 'col': 'colossians', 'co': 'colossians',
    '1 thessalonians': '1thessalonians', '1thessalonians': '1thessalonians', '1thess': '1thessalonians', '1th': '1thessalonians', '1thes': '1thessalonians',
    'first thessalonians': '1thessalonians', 'i thessalonians': '1thessalonians',
    '2 thessalonians': '2thessalonians', '2thessalonians': '2thessalonians', '2thess': '2thessalonians', '2th': '2thessalonians', '2thes': '2thessalonians',
    'second thessalonians': '2thessalonians', 'ii thessalonians': '2thessalonians',
    '1 timothy': '1timothy', '1timothy': '1timothy', '1tim': '1timothy', '1ti': '1timothy', '1t': '1timothy',
    'first timothy': '1timothy', 'i timothy': '1timothy',
    '2 timothy': '2timothy', '2timothy': '2timothy', '2tim': '2timothy', '2ti': '2timothy', '2t': '2timothy',
    'second timothy': '2timothy', 'ii timothy': '2timothy',
    'titus': 'titus', 'tit': 'titus', 'ti': 'titus',
    'philemon': 'philemon', 'philem': 'philemon', 'phm': 'philemon', 'pm': 'philemon',
    'hebrews': 'hebrews', 'heb': 'hebrews',
    'james': 'james', 'jas': 'james', 'jm': 'james',
    '1 peter': '1peter', '1peter': '1peter', '1pet': '1peter', '1pe': '1peter', '1pt': '1peter', '1p': '1peter',
    'first peter': '1peter', 'i peter': '1peter',
    '2 peter': '2peter', '2peter': '2peter', '2pet': '2peter', '2pe': '2peter', '2pt': '2peter', '2p': '2peter',
    'second peter': '2peter', 'ii peter': '2peter',
    '1 john': '1john', '1john': '1john', '1jn': '1john', '1jo': '1john', '1j': '1john',
    'first john': '1john', 'i john': '1john',
    '2 john': '2john', '2john': '2john', '2jn': '2john', '2jo': '2john', '2j': '2john',
    'second john': '2john', 'ii john': '2john',
    '3 john': '3john', '3john': '3john', '3jn': '3john', '3jo': '3john', '3j': '3john',
    'third john': '3john', 'iii john': '3john',
    'jude': 'jude', 'jud': 'jude', 'jd': 'jude',
    'revelation': 'revelation', 'rev': 'revelation', 're': 'revelation', 'the revelation': 'revelation',
}

# Reverse mapping for display
BOOK_DISPLAY_NAMES = {
    'genesis': 'Genesis', 'exodus': 'Exodus', 'leviticus': 'Leviticus', 
    'numbers': 'Numbers', 'deuteronomy': 'Deuteronomy', 'joshua': 'Joshua',
    'judges': 'Judges', 'ruth': 'Ruth', '1samuel': '1 Samuel', '2samuel': '2 Samuel',
    '1kings': '1 Kings', '2kings': '2 Kings', '1chronicles': '1 Chronicles', 
    '2chronicles': '2 Chronicles', 'ezra': 'Ezra', 'nehemiah': 'Nehemiah',
    'esther': 'Esther', 'job': 'Job', 'psalm': 'Psalms', 'proverbs': 'Proverbs',
    'ecclesiastes': 'Ecclesiastes', 'songofsolomon': 'Song of Solomon',
    'isaiah': 'Isaiah', 'jeremiah': 'Jeremiah', 'lamentations': 'Lamentations',
    'ezekiel': 'Ezekiel', 'daniel': 'Daniel', 'hosea': 'Hosea', 'joel': 'Joel',
    'amos': 'Amos', 'obadiah': 'Obadiah', 'jonah': 'Jonah', 'micah': 'Micah',
    'nahum': 'Nahum', 'habakkuk': 'Habakkuk', 'zephaniah': 'Zephaniah',
    'haggai': 'Haggai', 'zechariah': 'Zechariah', 'malachi': 'Malachi',
    'matthew': 'Matthew', 'mark': 'Mark', 'luke': 'Luke', 'john': 'John',
    'acts': 'Acts', 'romans': 'Romans', '1corinthians': '1 Corinthians',
    '2corinthians': '2 Corinthians', 'galatians': 'Galatians', 'ephesians': 'Ephesians',
    'philippians': 'Philippians', 'colossians': 'Colossians', '1thessalonians': '1 Thessalonians',
    '2thessalonians': '2 Thessalonians', '1timothy': '1 Timothy', '2timothy': '2 Timothy',
    'titus': 'Titus', 'philemon': 'Philemon', 'hebrews': 'Hebrews', 'james': 'James',
    '1peter': '1 Peter', '2peter': '2 Peter', '1john': '1 John', '2john': '2 John',
    '3john': '3 John', 'jude': 'Jude', 'revelation': 'Revelation'
}


def extract_bible_references(text):
    """
    Extract all Bible references from text
    """
    references = []
    
    # Pattern to match Bible references
    pattern = r'\b((?:(?:First|Second|Third|1|2|3|I|II|III)\s+)?[A-Za-z]+\.?)\s+(\d+)'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for book_raw, chapter in matches:
        # Clean up book name
        book_clean = book_raw.strip().rstrip('.').lower()
        book_clean = re.sub(r'\s+', ' ', book_clean)
        
        # Map to CSV format (lowercase, no spaces)
        if book_clean in BOOK_MAPPINGS:
            csv_book = BOOK_MAPPINGS[book_clean]
            references.append((csv_book, chapter))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_refs = []
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    
    return unique_refs


class BibleSearchEngine:
    def __init__(self):
        self.qwen_s_model = None
        self.cross_encoder_qnli = None # For QNLI tasks
        self.cross_encoder_stsb = None # For STS tasks
       
        # Qwen-0.6B question embeddings
        self.question_index_qwen_s = None
        self.question_embeddings_qwen_s = None
        self.question_df_qwen_s = None
       
        # Qwen-8B question embeddings
        self.question_embeddings_qwen = None
        self.question_df_qwen = None
       
        # Qwen-8B QnA embeddings
        self.qna_index_qwen = None
        self.qna_embeddings_qwen = None
        self.qna_df_qwen = None
       
        # Qwen-8B chapter embeddings
        self.chapters_index_qwen = None
        self.chapters_embeddings_qwen = None
        self.chapters_df_qwen = None
       
        # Qwen-8B verse embeddings
        self.verses_index_qwen = None
        self.verses_embeddings_qwen = None
        self.verses_df_qwen = None
       
    def load_resources(self):
        """Load all models and indices"""
        if self.qwen_s_model is None:
            self.qwen_s_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
           
        if self.cross_encoder_qnli is None:
            self.cross_encoder_qnli = CrossEncoder('cross-encoder/qnli-electra-base')
           
        if self.cross_encoder_stsb is None:
            self.cross_encoder_stsb = CrossEncoder('cross-encoder/stsb-roberta-large')
        # Load Qwen-0.6B question index
        if self.question_index_qwen_s is None:
            self.question_index_qwen_s = faiss.read_index(
                str(settings.VECTOR_DB_DIR / 'question_index_qwen3-embeddin_0_6b.faiss')
            )
            self.question_embeddings_qwen_s = np.load(
                settings.VECTOR_DB_DIR / 'question_embeddings_qwen3-embeddin_0_6b.npy'
            )
            self.question_df_qwen_s = pd.read_csv(
                settings.VECTOR_DB_DIR / 'question_metadata_qwen3-embeddin_0_6b.csv'
            )
           
        # Load Qwen-8B question embeddings
        if self.question_embeddings_qwen is None:
            self.question_embeddings_qwen = np.load(
                settings.VECTOR_DB_DIR / 'question_embeddings_qwen3_embedding_8b.npy'
            )
            self.question_df_qwen = pd.read_csv(
                settings.VECTOR_DB_DIR / 'question_metadata_qwen3_embedding_8b.csv'
            )
           
        # Load Qwen-8B QnA index
        if self.qna_index_qwen is None:
            self.qna_index_qwen = faiss.read_index(
                str(settings.VECTOR_DB_DIR / 'qna_index_qwen3_embedding_8b.faiss')
            )
            self.qna_embeddings_qwen = np.load(
                settings.VECTOR_DB_DIR / 'qna_embeddings_qwen3_embedding_8b.npy'
            )
            self.qna_df_qwen = pd.read_csv(
                settings.VECTOR_DB_DIR / 'qna_metadata_qwen3_embedding_8b.csv'
            )
           
        # Load Qwen-8B chapter index
        if self.chapters_index_qwen is None:
            self.chapters_index_qwen = faiss.read_index(
                str(settings.VECTOR_DB_DIR / 'chapters_index_qwen3_embedding_8b.faiss')
            )
            self.chapters_embeddings_qwen = np.load(
                settings.VECTOR_DB_DIR / 'chapters_embeddings_qwen3_embedding_8b.npy'
            )
            self.chapters_df_qwen = pd.read_csv(
                settings.VECTOR_DB_DIR / 'chapters_metadata_qwen3_embedding_8b.csv'
            )
           
        # Load Qwen-8B verse index
        if self.verses_index_qwen is None:
            self.verses_index_qwen = faiss.read_index(
                str(settings.VECTOR_DB_DIR / 'verses_index_qwen3_embedding_8b.faiss')
            )
            self.verses_embeddings_qwen = np.load(
                settings.VECTOR_DB_DIR / 'verses_embeddings_qwen3_embedding_8b.npy'
            )
            self.verses_df_qwen = pd.read_csv(
                settings.VECTOR_DB_DIR / 'verses_metadata_qwen3_embedding_8b.csv'
            )
   
    def normalize_embeddings(self, embeddings):
        """Normalize embeddings to unit length for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10 # Avoid division by zero
        return embeddings / norms
   
    def compute_cosine_similarity(self, query_embedding, target_embeddings):
        """Compute cosine similarity between query and target embeddings"""
        # Normalize both embeddings
        query_norm = self.normalize_embeddings(query_embedding)
        target_norm = self.normalize_embeddings(target_embeddings)
       
        # Compute cosine similarity
        similarities = np.dot(target_norm, query_norm.T).flatten()
        return similarities
   
    def search_with_faiss(self, query_embedding, index, k=5):
        """Search using FAISS index with normalized embeddings"""
        # Normalize the query embedding
        query_norm = self.normalize_embeddings(query_embedding)
       
        # Search with FAISS
        distances, indices = index.search(query_norm, k)
       
        # Convert distances to similarities (FAISS IndexFlatIP returns inner products)
        # Since embeddings are normalized, inner product = cosine similarity
        similarities = distances.flatten()
       
        return similarities, indices
   
    def cross_encoder_rerank(self, query, texts, cross_encoder_model, top_k=5):
        """Rerank texts using specified cross-encoder based on similarity to query"""
        if not texts:
            return [], []
           
        # Create pairs for cross-encoder - using the format (query, text)
        pairs = [(query, text) for text in texts]
       
        # Get scores from cross-encoder
        scores = cross_encoder_model.predict(pairs)
       
        # Sort by score in descending order
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]
        top_scores = scores[top_indices]
       
        return top_indices, top_scores
   
    def search(self, query):
       
        self.load_resources()
       
        # Step 1: Embed user query with qwen_s
        query_embedding_qwen_s = self.qwen_s_model.encode([query], normalize_embeddings=True)
       
        # Step 2: Find top 100 similar questions in qwen_s embeddings using FAISS
        similarities_qwen_s, indices_qwen_s = self.search_with_faiss(
            query_embedding_qwen_s,
            self.question_index_qwen_s,
            k=100
        )
       
        # Rerank top 100 questions with STSB cross-encoder to get top 5
        top_100_questions = self.question_df_qwen_s.iloc[indices_qwen_s[0]]['Question'].tolist()
        reranked_question_indices, reranked_question_scores = self.cross_encoder_rerank(
            query, top_100_questions, self.cross_encoder_stsb, top_k=5
        )
       
        # Get the top 5 question indices after reranking
        top_question_indices_qwen_s = indices_qwen_s[0][reranked_question_indices]
        top_question_indices_qwen = top_question_indices_qwen_s # Assuming same indexing
       
        # Print the top 5 corresponding Qwen question texts
        print("\n[Top 5 Qwen Matched Questions]")
        for i, idx in enumerate(top_question_indices_qwen):
            matched_question_text = self.question_df_qwen.iloc[idx]['Question']
            print(f"{i+1}. {matched_question_text}")
        # Average the embeddings of the top 5 questions for downstream use
        question_embeddings_top5 = self.question_embeddings_qwen[top_question_indices_qwen]
        question_embedding_qwen = np.mean(question_embeddings_top5, axis=0).reshape(1, -1)
        # Step 3: For each top 5 question, search top 20 QnA using its individual Qwen question embedding with FAISS
        all_top_qna_indices = []
        for top_question_idx in top_question_indices_qwen:
            question_embedding_this = self.question_embeddings_qwen[top_question_idx:top_question_idx+1]
            similarities_qna, indices_qna = self.search_with_faiss(
                question_embedding_this,
                self.qna_index_qwen,
                k=20
            )
           
            # Rerank top 20 QnA with QNLI cross-encoder using only the Question column
            top_20_qna_questions = self.qna_df_qwen.iloc[indices_qna[0]]['Question'].tolist()
            reranked_qna_indices, reranked_qna_scores = self.cross_encoder_rerank(
                query, top_20_qna_questions, self.cross_encoder_qnli, top_k=2
            )
           
            # Get top 2 QnA after reranking for this question
            top_qna_indices_this = indices_qna[0][reranked_qna_indices]
            all_top_qna_indices.extend(top_qna_indices_this)
       
        # Deduplicate QnA indices across all searches
        unique_top_qna_indices = list(set(all_top_qna_indices))
        top_qna = self.qna_df_qwen.iloc[unique_top_qna_indices]
        # Step 4: Extract all chapter references from top QnA (deduplicated)
        all_references = []
        for _, row in top_qna.iterrows():
            answer_text = str(row.get('Answer', ''))
            question_text = str(row.get('Question', ''))
            combined_text = answer_text + ' ' + question_text
            refs = extract_bible_references(combined_text)
            all_references.extend(refs)
       
        # Remove duplicates while preserving order
        unique_references = list(dict.fromkeys(all_references))
       
        if not unique_references:
            return []
       
        # Find matching chapter indices in database
        chapter_indices = []
        chapter_refs = []
        for book_csv, chapter_num in unique_references:
            matched = self.chapters_df_qwen[
                (self.chapters_df_qwen['book'] == book_csv) &
                (self.chapters_df_qwen['chapter'] == int(chapter_num))
            ]
            if not matched.empty:
                chapter_indices.append(matched.index[0])
                chapter_refs.append((book_csv, int(chapter_num)))
       
        if not chapter_indices:
            return []
       
        # Step 5: Get top 10 chapters using averaged Qwen embedding
        chapter_embeddings_subset = self.chapters_embeddings_qwen[chapter_indices]
       
        # Calculate similarities between chapters and the averaged Qwen question embedding
        chapter_similarities = self.compute_cosine_similarity(
            question_embedding_qwen,
            chapter_embeddings_subset
        )
       
        # Get top 10 chapters by Qwen similarity
        top_10_chapter_indices = np.argsort(chapter_similarities)[::-1][:10]
        top_10_chapter_original_indices = [chapter_indices[idx] for idx in top_10_chapter_indices]
        top_10_chapter_refs = [chapter_refs[idx] for idx in top_10_chapter_indices]
       
        # Rerank top 10 chapters with QNLI cross-encoder using chapter text
        top_10_chapter_texts = []
        for idx in top_10_chapter_original_indices:
            chapter_data = self.chapters_df_qwen.iloc[idx]
            book_display = BOOK_DISPLAY_NAMES.get(chapter_data['book'], chapter_data['book'].title())
            chapter_text = f"{book_display} Chapter {chapter_data['chapter']}: {chapter_data['text']}"
            top_10_chapter_texts.append(chapter_text)
       
        reranked_chapter_indices, reranked_chapter_scores = self.cross_encoder_rerank(
            query, top_10_chapter_texts, self.cross_encoder_qnli, top_k=10
        )
       
        # Step 6: For each of the top 10 reranked chapters, get all verses and identify top 3 for highlighting
        results = []
        for reranked_idx in reranked_chapter_indices:
            original_idx = top_10_chapter_original_indices[reranked_idx]
            chapter_data = self.chapters_df_qwen.iloc[original_idx]
           
            book_csv = chapter_data['book']
            chapter_num = int(chapter_data['chapter'])
            book_display = BOOK_DISPLAY_NAMES.get(book_csv, book_csv.title())
           
            # Get ALL verses for this chapter (sorted by verse number)
            chapter_verses = self.verses_df_qwen[
                (self.verses_df_qwen['book'] == book_csv) &
                (self.verses_df_qwen['chapter'] == chapter_num)
            ].sort_values('verse').copy()
           
            if chapter_verses.empty:
                continue
           
            # Get verse embeddings and calculate similarities for ALL verses
            verse_indices = chapter_verses.index.tolist()
            verse_embeddings = self.verses_embeddings_qwen[verse_indices]
           
            verse_similarities = self.compute_cosine_similarity(
                question_embedding_qwen,
                verse_embeddings
            )
           
            # Add similarities to the dataframe
            chapter_verses['similarity'] = verse_similarities
           
            # Get top 10 verses by Qwen similarity for reranking
            top_10_verse_indices = np.argsort(verse_similarities)[::-1][:10]
            top_10_verse_original_indices = [verse_indices[idx] for idx in top_10_verse_indices]
           
            # Rerank top 10 verses with QNLI cross-encoder using verse text
            top_10_verse_texts = self.verses_df_qwen.iloc[top_10_verse_original_indices]['text'].tolist()
            reranked_verse_indices, reranked_verse_scores = self.cross_encoder_rerank(
                query, top_10_verse_texts, self.cross_encoder_qnli, top_k=3
            )
           
            # Get top 3 verses after reranking for highlighting
            top_3_verse_indices = [top_10_verse_original_indices[idx] for idx in reranked_verse_indices]
            top_3_verses = self.verses_df_qwen.iloc[top_3_verse_indices]
           
            # Compute mean similarity: mean of chapter similarity and top 3 verse similarities
            top_3_similarities = [chapter_verses.loc[idx]['similarity'] for idx in top_3_verse_indices]
            cosine_sim = chapter_similarities[chapter_indices.index(original_idx)]
            mean_sim = np.mean([cosine_sim] + top_3_similarities)
           
            # Create a mapping of verse numbers to cross-encoder scores for highlighting
            highlight_scores = {}
            for i, (_, verse_row) in enumerate(top_3_verses.iterrows()):
                highlight_scores[int(verse_row['verse'])] = float(reranked_verse_scores[i])
           
            # Prepare ALL verses data for display
            verses_data = []
            for _, verse_row in chapter_verses.iterrows():
                verse_num = int(verse_row['verse'])
                is_highlighted = verse_num in highlight_scores
               
                verses_data.append({
                    'verse': verse_num,
                    'text': verse_row['text'],
                    'is_highlighted': is_highlighted,
                    'highlight_score': highlight_scores.get(verse_num, 0.0),
                    'similarity': float(verse_row['similarity'])
                })
            chapter_idx = top_10_chapter_original_indices[reranked_idx]
            results.append({
                'book': book_display,
                'chapter': chapter_num,
                'chapter_id': f"{book_display} {chapter_num}",
               
                'similarity': float(mean_sim),
                'verses': verses_data,
                'highlighted_verses': [int(v) for v in highlight_scores.keys()] # For reference
            })
       
        return results