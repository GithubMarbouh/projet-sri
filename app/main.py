from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel
from .indexer import DocumentIndexer
from .search import DocumentSearcher

app = FastAPI(title="Moteur de Recherche en Droit Commercial")

# Initialisation des classes
indexer = DocumentIndexer()
searcher = DocumentSearcher()

class SearchResult(BaseModel):
    title: str
    path: str
    score: float

@app.post("/index")
async def index_documents():
    """Endpoint pour indexer tous les documents."""
    try:
        indexer.index_all_documents()
        return {"message": "Indexation terminée avec succès"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/search", response_model=List[SearchResult])
async def search(
    query: str,
    search_type: str = Query("hybrid", enum=["keyword", "semantic", "hybrid"]),
    limit: Optional[int] = 10
):
    """
    Endpoint pour effectuer une recherche.
    
    - query: Texte de la requête
    - search_type: Type de recherche (keyword, semantic, ou hybrid)
    - limit: Nombre maximum de résultats
    """
    try:
        if search_type == "keyword":
            results = searcher.keyword_search(query, limit)
        elif search_type == "semantic":
            results = searcher.semantic_search(query, limit)
        else:
            results = searcher.hybrid_search(query, limit)
        
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)