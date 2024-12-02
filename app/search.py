from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import Term
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

class DocumentSearcher:
    def __init__(self, index_dir="index"):
        self.ix = index.open_dir(index_dir)
        self.bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def keyword_search(self, query_text, limit=10):
        """Recherche par mots-clés utilisant Whoosh."""
        with self.ix.searcher() as searcher:
            query_parser = QueryParser("content", self.ix.schema)
            query = query_parser.parse(query_text)
            results = searcher.search(query, limit=limit)
            
            return [{
                'title': result['title'],
                'path': result['path'],
                'score': result.score,
            } for result in results]

    def semantic_search(self, query_text, limit=10):
        """Recherche sémantique utilisant BERT."""
        query_embedding = self.bert_model.encode(query_text)
        
        results = []
        with self.ix.searcher() as searcher:
            for doc in searcher.all_stored_fields():
                doc_vector = eval(doc['vector'])
                similarity = 1 - cosine(query_embedding, doc_vector)
                results.append({
                    'title': doc['title'],
                    'path': doc['path'],
                    'score': similarity
                })
        
        # Trier par score de similarité
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def hybrid_search(self, query_text, limit=10, w1=0.5, w2=0.5):
        """Recherche hybride combinant mots-clés et sémantique."""
        keyword_results = self.keyword_search(query_text, limit=limit)
        semantic_results = self.semantic_search(query_text, limit=limit)
        
        # Combiner et normaliser les scores
        combined_results = {}
        for result in keyword_results:
            combined_results[result['path']] = {
                'title': result['title'],
                'path': result['path'],
                'keyword_score': result['score'],
                'semantic_score': 0
            }
            
        for result in semantic_results:
            if result['path'] in combined_results:
                combined_results[result['path']]['semantic_score'] = result['score']
            else:
                combined_results[result['path']] = {
                    'title': result['title'],
                    'path': result['path'],
                    'keyword_score': 0,
                    'semantic_score': result['score']
                }
        
        # Calculer le score final
        final_results = []
        for doc_info in combined_results.values():
            final_score = (w1 * doc_info['keyword_score'] + 
                         w2 * doc_info['semantic_score'])
            final_results.append({
                'title': doc_info['title'],
                'path': doc_info['path'],
                'score': final_score
            })
        
        # Trier par score final
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:limit]