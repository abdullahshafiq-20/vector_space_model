import os
root = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(root, 'nltk_data')
os.makedirs(download_dir, exist_ok=True)

import nltk
nltk.data.path.append(download_dir)
try:
    nltk.download('punkt', download_dir=download_dir)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

import re
import json
import math
import numpy as np
from collections import defaultdict
import datetime
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class VectorSpaceModel:
    def __init__(self):
        self.totalNumberOfDocuments = 50
        self.documentsPath = "Abstracts/{}.txt"
        self.stopWordFileName = "./stop_words.txt"
        self.indexFileName = "./static/index.txt"
        self.modelFileName = "./vsm_model.json"
        
        os.makedirs("./static", exist_ok=True)
        abstracts_dir = os.path.dirname(self.documentsPath.format(''))
        os.makedirs(abstracts_dir, exist_ok=True)
        
        self.stemmer = PorterStemmer()
        self.stop_words = self.getStopWordsFromFile()
        self.vocab = {}  # term -> index in vector
        self.doc_ids = {}  # doc_id -> filename
        self.doc_terms = defaultdict(dict)  # doc_id -> {term -> term_freq}
        self.term_doc_freq = defaultdict(int)  # term -> document frequency
        self.idf = {}  # term -> idf value
        self.tfidf_matrix = None
        
        self._build_model()

    def getStopWordsFromFile(self):
        """Returns stop words from file"""
        stop_words = set()
        try:
            with open(self.stopWordFileName, 'r') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stop_words.add(word)
            return stop_words
        except FileNotFoundError:
            print(f"Warning: Stop words file {self.stopWordFileName} not found.")
            return set()

    def _preprocess_text(self, text):
        """Tokenize, remove stop words and apply stemming to the text"""
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        preprocessed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                stemmed_token = self.stemmer.stem(token)
                preprocessed_tokens.append(stemmed_token)
        return preprocessed_tokens

    def _read_document(self, file_path):
        """Read document with proper encoding handling"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')

    def _build_model(self):
        """Build the Vector Space Model or load it if already exists"""
        print("Building Vector Space Model...")
        
        if os.path.exists(self.modelFileName):
            try:
                self._load_model()
                print("Loaded existing Vector Space Model.")
                return
            except Exception as e:
                print(f"Error loading saved model: {e}")
                print("Building model from scratch...")
        
        abstracts_dir = os.path.dirname(self.documentsPath.format(''))
        try:
            available_docs = [f for f in os.listdir(abstracts_dir) if f.endswith('.txt')]
            self.totalNumberOfDocuments = len(available_docs)
            print(f"Found {self.totalNumberOfDocuments} documents in {abstracts_dir}")
        except FileNotFoundError:
            print(f"Abstracts directory not found. Creating empty directory.")
            os.makedirs(abstracts_dir, exist_ok=True)
            available_docs = []
        
        processed_docs = 0
        failed_docs = []
        
        print("Processing documents...")
        for doc_file in available_docs:
            try:
                # Extract document ID from filename (removing .txt extension)
                doc_id = int(os.path.splitext(doc_file)[0])
                file_path = os.path.join(abstracts_dir, doc_file)
                
                content = self._read_document(file_path)
                
                # Process document
                self.doc_ids[doc_id-1] = doc_file
                tokens = self._preprocess_text(content)
                
                # Count term frequency in document
                term_freq = {}
                for term in tokens:
                    term_freq[term] = term_freq.get(term, 0) + 1
                
                # Store term frequencies for this document
                self.doc_terms[doc_id-1] = term_freq
                
                # Update document frequency for each unique term
                for term in set(tokens):
                    self.term_doc_freq[term] += 1
                    # Add term to vocabulary if new
                    if term not in self.vocab:
                        self.vocab[term] = len(self.vocab)
                
                processed_docs += 1
                
            except Exception as e:
                failed_docs.append((doc_file, str(e)))
        
        # Calculate IDF for each term
        print("Calculating IDF values...")
        total_doc_count = max(processed_docs, 1)  # Avoid division by zero
        for term, doc_freq in self.term_doc_freq.items():
            self.idf[term] = math.log(total_doc_count / (1 + doc_freq), 10)
        
        # Build the TF-IDF matrix
        print("Building TF-IDF matrix...")
        self._build_tfidf_matrix()
        
        # Create index file
        self._create_index_file()
        
        # Log processing results
        print(f"Successfully processed: {processed_docs} documents")
        print(f"Vocabulary size: {len(self.vocab)}")
        if failed_docs:
            print("\nFailed documents:")
            for doc, error in failed_docs:
                print(f"- {doc}: {error}")
        
        self._save_model()

    def _create_index_file(self):
        """Create an index file with terms and their document frequencies"""
        print("Creating index file...")
        os.makedirs(os.path.dirname(self.indexFileName), exist_ok=True)
        
        # Sort terms by document frequency in descending order
        sorted_terms = sorted(self.term_doc_freq.items(), key=lambda x: x[1], reverse=True)
        
        with open(self.indexFileName, 'w', encoding='utf-8') as f:
            f.write("TERM\tDOCUMENT FREQUENCY\tDOCUMENTS\n")
            for term, doc_freq in sorted_terms:
                # Find all documents containing this term
                docs = [str(doc_id+1) for doc_id, terms in self.doc_terms.items() if term in terms]
                f.write(f"{term}\t{doc_freq}\t{', '.join(docs)}\n")
        
        print(f"Index file created at {self.indexFileName}")

    def _build_tfidf_matrix(self):
        """Build the TF-IDF matrix for documents"""
        n_docs = max(len(self.doc_terms), 1)  # Use actual document count
        n_terms = len(self.vocab)
        
        # Create sparse matrix for efficiency
        rows = []
        cols = []
        data = []
        
        for doc_id, term_freqs in self.doc_terms.items():
            for term, freq in term_freqs.items():
                term_idx = self.vocab[term]
                # Raw term frequency
                tf = freq  
                # Standard IDF: log(N/df)
                tfidf = tf * self.idf[term]
                
                rows.append(doc_id)
                cols.append(term_idx)
                data.append(tfidf)
        
        self.tfidf_matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms))

    def _save_model(self):
        """Save the Vector Space Model to disk"""
        print("Saving Vector Space Model...")
        
        # Convert sparse matrix to list format for JSON serialization
        matrix_data = {
            'data': self.tfidf_matrix.data.tolist(),
            'indices': self.tfidf_matrix.indices.tolist(),
            'indptr': self.tfidf_matrix.indptr.tolist(),
            'shape': self.tfidf_matrix.shape
        }
        
        model_data = {
            'metadata': {
                'total_documents': self.totalNumberOfDocuments,
                'total_terms': len(self.vocab),
                'created_at': str(datetime.datetime.now())
            },
            'document_mapping': {
                str(doc_id): {
                    'filename': filename
                }
                for doc_id, filename in self.doc_ids.items()
            },
            'vocabulary': {term: idx for term, idx in self.vocab.items()},
            'idf': {term: float(idf_val) for term, idf_val in self.idf.items()},
            'tfidf_matrix': matrix_data
        }
        
        try:
            # Ensure model file directory exists (should be current directory)
            model_path = os.path.abspath(self.modelFileName)
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False)
            print(f"Model saved successfully at {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Print more debugging info
            print(f"Attempted to save to: {os.path.abspath(self.modelFileName)}")
            print(f"Current directory: {os.getcwd()}")

    def _load_model(self):
        """Load the Vector Space Model from disk"""
        with open(self.modelFileName, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # Load document mapping
        self.doc_ids = {
            int(doc_id): info['filename']
            for doc_id, info in model_data['document_mapping'].items()
        }
        
        # Load vocabulary and IDF values
        self.vocab = {term: int(idx) for term, idx in model_data['vocabulary'].items()}
        self.idf = {term: float(idf) for term, idf in model_data['idf'].items()}
        
        # Load TF-IDF matrix
        matrix_data = model_data['tfidf_matrix']
        self.tfidf_matrix = csr_matrix(
            (matrix_data['data'], matrix_data['indices'], matrix_data['indptr']),
            shape=tuple(matrix_data['shape'])
        )
    
    def createQueryVector(self, query_terms):
        """Create vector for query terms"""
        # Preprocess query terms
        processed_terms = []
        for term in query_terms:
            if term not in self.stop_words:
                processed_terms.append(self.stemmer.stem(term.lower()))
        
        # Create query vector with same dimensions as document vectors
        query_vector = np.zeros((1, len(self.vocab)))
        
        # Calculate query term frequencies
        query_term_freq = {}
        for term in processed_terms:
            query_term_freq[term] = query_term_freq.get(term, 0) + 1
        
        # Apply TF-IDF weighting to query
        for term, freq in query_term_freq.items():
            if term in self.vocab:  # Only consider terms in our vocabulary
                term_idx = self.vocab[term]
                tf = freq  # Raw term frequency
                idf = self.idf.get(term, 0)  # Use 0 if term not in corpus
                query_vector[0, term_idx] = tf * idf
        
        # Normalize query vector (L2 norm)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
            
        return query_vector
    
    def executeQuery(self, query):
        """Process a query and return relevant documents with cosine similarity above threshold"""
        try:
            query_parts = nltk.word_tokenize(query)
        except LookupError:
            print("Warning: NLTK punkt tokenizer not available. Using simple tokenization.")
            query_parts = query.split()
        
        try:
            alpha_threshold = float(query_parts[-1])
            query_terms = query_parts[:-1]
        except (ValueError, IndexError):
            print("Warning: Invalid threshold. Using default threshold of 0.5")
            alpha_threshold = 0.001
            query_terms = query_parts
        
        # Get query vector
        query_vector = self.createQueryVector(query_terms)
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get documents above threshold, sorted by similarity
        results = []
        for doc_id, similarity in enumerate(similarities):
            if similarity >= alpha_threshold:
                # Use the actual document ID (from doc_ids if available, or doc_id+1)
                doc_num = doc_id + 1  # Default to 1-based indexing
                doc_name = self.doc_ids.get(doc_id, f"{doc_num}.txt")
                results.append((similarity, int(doc_name.split('.')[0])))  # Extract document number
        
        # Sort by similarity score in descending order
        results.sort(reverse=True)
        
        return results
    

if __name__ == "__main__":
    try:
        vsm = VectorSpaceModel()
        print("\nVector Space Model built successfully!")
        
        # Example query - use a simple query with common terms and low threshold to get results
        query = "deep"
        print(f"\nExecuting query: '{query}'")
        
        results = vsm.executeQuery(query)
        
        if results:
            print(f"\nQuery Results (found {len(results)} documents):")
            for score, doc_id in results:
                print(f"Document ID: {doc_id}, Similarity Score: {score:.4f}")
        else:
            print("\nNo results found matching the query and threshold.")
            print("Try lowering the threshold or using different query terms.")
    except Exception as e:
        print(f"\nError running Vector Space Model: {e}")
        import traceback
        traceback.print_exc()
