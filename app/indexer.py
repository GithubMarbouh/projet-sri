import os
import PyPDF2
import spacy
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

class DocumentIndexer:
    def __init__(self, index_dir="index", data_dir="data/raw"):
        self.index_dir = index_dir
        self.data_dir = data_dir
        self.nlp = spacy.load("fr_core_news_md")
        self.bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        # Création du schéma Whoosh
        self.schema = Schema(
            path=ID(stored=True),
            title=TEXT(stored=True),
            content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            vector=TEXT(stored=True)
        )
        
        # Création du répertoire d'index s'il n'existe pas
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            self.ix = index.create_in(index_dir, self.schema)
        else:
            self.ix = index.open_dir(index_dir)

    def extract_text_from_pdf(self, pdf_path):
        """Extraire le texte d'un fichier PDF."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Erreur lors de la lecture du PDF {pdf_path}: {e}")
        return text

    def preprocess_text(self, text):
        """Prétraiter le texte avec spaCy."""
        doc = self.nlp(text)
        # Suppression des stop words et des tokens non alphabétiques
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    def create_bert_embedding(self, text):
        """Créer un embedding BERT pour le texte."""
        embedding = self.bert_model.encode(text[:512])  # Limitation à 512 tokens
        return embedding.tolist()

    def index_document(self, pdf_path):
        """Indexer un document PDF."""
        try:
            # Extraction du texte
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"Aucun texte extrait de {pdf_path}")
                return

            # Prétraitement
            processed_text = self.preprocess_text(text)
            
            # Création de l'embedding
            embedding = self.create_bert_embedding(text)
            
            # Ajout à l'index
            writer = self.ix.writer()
            writer.add_document(
                path=str(pdf_path),
                title=os.path.basename(pdf_path),
                content=processed_text,
                vector=str(embedding)
            )
            writer.commit()
            print(f"Document indexé avec succès: {pdf_path}")
            
        except Exception as e:
            print(f"Erreur lors de l'indexation de {pdf_path}: {e}")

    def index_all_documents(self):
        """Indexer tous les documents PDF du répertoire data."""
        pdf_files = Path(self.data_dir).glob("**/*.pdf")
        for pdf_path in pdf_files:
            self.index_document(str(pdf_path))