import streamlit as st
import requests
import os
import PyPDF2
from pathlib import Path

# Configuration de l'interface
st.set_page_config(
    page_title="Moteur de Recherche en Droit Commercial",
    page_icon="⚖️",
    layout="wide"
)

# URL de l'API
API_URL = "http://localhost:8000"

def main():
    st.title("⚖️ Moteur de Recherche en Droit Commercial")
    
    # Sidebar pour les options
    with st.sidebar:
        st.header("Options de recherche")
        search_type = st.selectbox(
            "Type de recherche",
            ["hybrid", "keyword", "semantic"],
            help="Choisissez la méthode de recherche"
        )
        
        limit = st.slider(
            "Nombre de résultats",
            min_value=1,
            max_value=20,
            value=10
        )
        
        if st.button("Réindexer les documents"):
            with st.spinner("Indexation en cours..."):
                response = requests.post(f"{API_URL}/index")
                if response.status_code == 200:
                    st.success("Indexation terminée avec succès!")
                else:
                    st.error("Erreur lors de l'indexation")
    
    # Zone de recherche principale
    query = st.text_input("Entrez votre requête juridique")
    
    if query:
        with st.spinner("Recherche en cours..."):
            try:
                response = requests.get(
                    f"{API_URL}/search",
                    params={
                        "query": query,
                        "search_type": search_type,
                        "limit": limit
                    }
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            with st.expander(f"{i}. {result['title']} (Score: {result['score']:.4f})"):
                                st.write(f"Chemin du fichier: {result['path']}")
                                
                                # Ajout d'un bouton pour visualiser le PDF
                                if os.path.exists(result['path']):
                                    try:
                                        with open(result['path'], 'rb') as pdf_file:
                                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                                            text = ""
                                            for page in pdf_reader.pages:
                                                text += page.extract_text() + "\n"
                                            st.text_area("Aperçu du contenu:", text[:1000] + "...", height=200)
                                    except Exception as e:
                                        st.error(f"Erreur lors de la lecture du PDF: {e}")
                    else:
                        st.info("Aucun résultat trouvé")
                        
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API: {e}")

if __name__ == "__main__":
    main()