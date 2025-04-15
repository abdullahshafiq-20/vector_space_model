import streamlit as st
from vsm import VectorSpaceModel
import os

class VSMWebUI:
    def __init__(self):
        self.vsm = VectorSpaceModel()
        self.setup_ui()

    def setup_ui(self):
        # Set page config
        st.set_page_config(
            page_title="Vector Space Model",
            page_icon="📊",
            layout="wide"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stSelectbox {
            border-radius: 6px;
        }
        .result-card {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
        }
        .similarity-score {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2980b9;
        }
        .document-id {
            font-size: 1.3rem;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("Vector Space Model Search Engine")
        st.markdown("Find documents by semantic similarity")
        st.markdown("---")

        # Search interface
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 🔎 Search Query")
            query = st.text_input(
                "Enter your search query...",
                placeholder="Example: information retrieval system"
            )
            threshold = st.number_input(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.001,
                step=0.001,
                format="%.3f",
                help="Minimum similarity score for returned documents"
                )
            st.markdown("#### 🎚️ Similarity Threshold")


            st.caption(f"Current threshold: `{threshold:.3f}` (lower = more results, higher = stricter match)")
        
        with col2:
            st.markdown("### How Vector Space Model Works")
            st.info("""
            📊 **Vector Space Model** represents documents and queries as vectors in a 
            multi-dimensional space where:
            
            - Each dimension corresponds to a unique term
            - Document similarity is measured using cosine similarity
            - Results are ranked by relevance to your query
            
            **Adjust the threshold** to control how many results you see.
            """)
        
        # Search button
        if st.button("Search", type="primary"):
            if not query:
                st.warning("Please enter a search query")
            else:
                try:
                    # Add threshold to the query for the VSM
                    full_query = f"{query} {threshold}"
                    
                    # Execute the query
                    results = self.vsm.executeQuery(full_query)
                    
                    if results:
                        st.success(f"Found {len(results)} relevant documents")
                        
                        # Display model statistics
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents in Index", len(self.vsm.doc_ids))
                        with col2:
                            st.metric("Terms in Vocabulary", len(self.vsm.vocab))
                        with col3:
                            st.metric("Results Found", len(results))
                        
                        # Results section
                        st.markdown("## Search Results")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["Card View", "Table View"])
                        
                        with tab1:
                            # Card view of results
                            for i, (score, doc_id) in enumerate(results):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <span class="document-id">Document {doc_id}</span>
                                        <p>Rank: #{i+1} | <span class="similarity-score">Similarity: {score:.4f}</span></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Load document content if available
                                    try:
                                        doc_path = f"Abstracts/{doc_id}.txt"
                                        if os.path.exists(doc_path):
                                            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                                                content = f.read()
                                                with st.expander(f"View Document Content"):
                                                    st.text(content)
                                    except Exception as e:
                                        st.warning(f"Could not load document content: {e}")
                        
                        with tab2:
                            # Table view of results
                            table_data = []
                            for i, (score, doc_id) in enumerate(results):
                                table_data.append({
                                    "Rank": i+1,
                                    "Document ID": doc_id,
                                    "Similarity Score": f"{score:.4f}"
                                })
                            st.table(table_data)
                    
                    else:
                        st.warning("No documents matched your query with the current threshold.")
                        st.info("Try lowering the similarity threshold or using different search terms.")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

def main():
    VSMWebUI()

if __name__ == "__main__":
    main()