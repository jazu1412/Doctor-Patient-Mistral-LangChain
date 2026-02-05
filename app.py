import streamlit as st
from mistralai import Mistral
import chromadb
from typing import List, Dict
import os
from dotenv import load_dotenv
from database import (
    sync_init_database,
    sync_sync_doctors,
    sync_get_available_doctors,
    sync_check_doctor_availability,
    sync_book_doctor,
    sync_release_doctor,
    sync_get_all_doctors
)

# Load environment variables
load_dotenv()

# Configuration - Load from environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "patient-doctor")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-small-latest")

# Validate required environment variables
required_vars = {
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "CHROMA_API_KEY": CHROMA_API_KEY,
    "CHROMA_TENANT": CHROMA_TENANT
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")
    st.stop()

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize Mistral and ChromaDB clients"""
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    chroma_client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Initialize database (silently fail if DB unavailable)
    try:
        sync_init_database()
    except Exception:
        # Silently fail - app will work without database for demo
        pass
    
    return mistral_client, collection

# Initialize session state
if 'clients_initialized' not in st.session_state:
    st.session_state.clients_initialized = False

try:
    mistral_client, collection = init_clients()
    st.session_state.clients_initialized = True
except Exception as e:
    st.error(f"Failed to initialize clients: {str(e)}")
    st.stop()

def get_symptom_embedding(symptoms: str) -> List[float]:
    """Convert patient symptoms to embedding using Mistral"""
    try:
        embeddings_response = mistral_client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[symptoms]
        )
        return embeddings_response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None

def find_best_doctor(symptoms_embedding: List[float], top_k: int = 3) -> List[Dict]:
    """Find the best matching doctors using vector similarity search, filtered by availability"""
    try:
        # Get more results than needed to account for unavailable doctors
        results = collection.query(
            query_embeddings=[symptoms_embedding],
            n_results=top_k * 3,  # Get more to filter by availability
            include=["documents", "metadatas", "distances"]
        )
        
        doctors = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doctor_info = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'doctor_name': results['metadatas'][0][i].get('doctor_name', 'N/A'),
                    'speciality': results['metadatas'][0][i].get('speciality', 'N/A'),
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                doctors.append(doctor_info)
        
        # Sync doctors to database (silently fail if DB unavailable)
        if doctors:
            try:
                sync_sync_doctors(doctors)
            except Exception:
                # Silently fail - don't show error during demo
                pass
        
        # Filter by availability - only return available doctors
        if doctors:
            doctor_names = [doc['doctor_name'] for doc in doctors]
            try:
                available_names = sync_get_available_doctors(doctor_names)
                if available_names:  # Only filter if we got results
                    doctors = [doc for doc in doctors if doc['doctor_name'] in available_names]
                # If no available names returned, show all (fail-safe)
            except Exception:
                # Silently fail - show all doctors if DB unavailable
                pass
        
        # Return only top_k available doctors
        return doctors[:top_k]
    except Exception as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return []

def get_doctor_recommendation(symptoms: str) -> str:
    """Get AI-powered recommendation explanation using Mistral chat"""
    try:
        prompt = f"""Based on the patient's symptoms: "{symptoms}", 
provide a brief explanation of why this doctor and speciality would be a good match. 
Keep the response concise (2-3 sentences)."""
        
        chat_response = mistral_client.chat.complete(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.warning(f"Could not generate AI recommendation: {str(e)}")
        return ""

# Streamlit UI
st.set_page_config(
    page_title="Doctor-Patient Matching System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Doctor-Patient Matching System")
st.markdown("Enter your symptoms below to find the best matching doctor for your needs.")

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    symptoms_input = st.text_area(
        "Describe your symptoms:",
        height=150,
        placeholder="e.g., I have been experiencing persistent headaches, dizziness, and blurred vision for the past week..."
    )

with col2:
    st.markdown("### Search Options")
    top_k = st.slider("Number of doctors to show", 1, 5, 3)
    show_ai_recommendation = st.checkbox("Show AI recommendation", value=True)

# Search button
if st.button("🔍 Find Doctor", type="primary", use_container_width=True):
    if not symptoms_input.strip():
        st.warning("Please enter your symptoms before searching.")
    else:
        with st.spinner("Analyzing your symptoms and finding the best doctor match..."):
            # Step 1: Convert symptoms to embedding
            symptoms_embedding = get_symptom_embedding(symptoms_input)
            
            if symptoms_embedding:
                # Step 2: Find best matching doctors
                doctors = find_best_doctor(symptoms_embedding, top_k=top_k)
                
                if doctors:
                    st.success(f"Found {len(doctors)} matching doctor(s)!")
                    
                    # Display results
                    for idx, doctor in enumerate(doctors, 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"### 👨‍⚕️ Doctor {idx}: {doctor['doctor_name']}")
                                st.markdown(f"**Speciality:** {doctor['speciality']}")
                                st.markdown(f"**Match Score:** {doctor['similarity_score']:.2%}")
                                
                                # Check availability status
                                try:
                                    is_available = sync_check_doctor_availability(doctor['doctor_name'])
                                    if is_available:
                                        st.success("✅ Available")
                                        # Booking button
                                        if st.button(f"📅 Book Appointment", key=f"book_{doctor['doctor_name']}_{idx}"):
                                            try:
                                                if sync_book_doctor(doctor['doctor_name']):
                                                    st.success(f"✅ Successfully booked appointment with {doctor['doctor_name']}!")
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Doctor is no longer available. Please search again.")
                                            except Exception:
                                                st.info("ℹ️ Booking service temporarily unavailable")
                                    else:
                                        st.warning("⏸️ Currently Unavailable")
                                except Exception:
                                    # If DB unavailable, show as available (fail-safe for demo)
                                    st.success("✅ Available")
                                    if st.button(f"📅 Book Appointment", key=f"book_{doctor['doctor_name']}_{idx}"):
                                        st.info("ℹ️ Booking feature requires database connection")
                                
                                if show_ai_recommendation and idx == 1:
                                    recommendation = get_doctor_recommendation(symptoms_input)
                                    if recommendation:
                                        with st.expander("💡 AI Recommendation", expanded=True):
                                            st.write(recommendation)
                            
                            with col2:
                                match_color = "green" if doctor['similarity_score'] > 0.7 else "orange" if doctor['similarity_score'] > 0.5 else "red"
                                st.markdown(
                                    f'<div style="text-align: center; padding: 20px; background-color: {match_color}20; border-radius: 10px; border: 2px solid {match_color};">'
                                    f'<h2 style="color: {match_color}; margin: 0;">{doctor["similarity_score"]:.0%}</h2>'
                                    f'<p style="margin: 5px 0 0 0; color: {match_color};">Match</p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            
                            st.divider()
                    
                    # Show selected doctor prominently
                    if len(doctors) > 0:
                        best_doctor = doctors[0]
                        st.markdown("---")
                        st.markdown("### ✅ Recommended Doctor")
                        st.info(
                            f"**{best_doctor['doctor_name']}** - {best_doctor['speciality']}\n\n"
                            f"Based on your symptoms, this doctor has the highest match score ({best_doctor['similarity_score']:.2%})."
                        )
                else:
                    st.error("No matching doctors found. Please try rephrasing your symptoms.")
            else:
                st.error("Failed to process your symptoms. Please try again.")

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Enter Symptoms**: Describe your symptoms in detail
    2. **AI Analysis**: Your symptoms are converted to embeddings
    3. **Vector Search**: We search for doctors with matching specialities
    4. **Best Match**: The system recommends the most suitable doctor
    
    The matching is based on semantic similarity between your symptoms 
    and doctor specialities using advanced AI embeddings.
    """)
    
    st.markdown("### 📊 System Info")
    st.code(f"""
    Model: {CHAT_MODEL}
    Embedding: {EMBEDDING_MODEL}
    Database: {CHROMA_DATABASE}
    """)
    
    if st.button("🔄 Refresh Connection"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🗄️ Database Management")
    
    if st.button("📊 View All Doctors"):
        try:
            all_doctors = sync_get_all_doctors()
            if all_doctors:
                import pandas as pd
                df = pd.DataFrame(all_doctors)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No doctors in database yet.")
        except Exception as e:
            st.error(f"Error fetching doctors: {str(e)}")
    
    if st.button("🔄 Sync Doctors from ChromaDB"):
        try:
            # Get all doctors from ChromaDB
            all_results = collection.get()
            doctors_to_sync = []
            if all_results.get('ids'):
                for i, doc_id in enumerate(all_results['ids']):
                    doctor_info = {
                        'id': doc_id,
                        'doctor_name': all_results['metadatas'][i].get('doctor_name', 'N/A'),
                        'speciality': all_results['metadatas'][i].get('speciality', 'N/A')
                    }
                    doctors_to_sync.append(doctor_info)
            
            sync_sync_doctors(doctors_to_sync)
            st.success(f"✅ Synced {len(doctors_to_sync)} doctors to database!")
        except Exception as e:
            st.error(f"Error syncing doctors: {str(e)}")


