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
from patient_processor import (
    process_patient_dataset,
    search_similar_cases,
    get_collection_stats,
    get_patient_collection
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home - Doctor Matching", "🔬 Clinical Decision Support", "⚙️ Admin Panel"],
    index=0
)

# Main content based on selected page
if page == "🏠 Home - Doctor Matching":
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

elif page == "🔬 Clinical Decision Support":
    st.title("🔬 Clinical Decision Support")
    st.markdown("Search for similar past patient cases to aid in clinical decision making.")
    
    # Get collection stats
    stats = get_collection_stats()
    if stats['success']:
        st.info(f"📊 Patient Cases Database: {stats['count']} cases available")
    else:
        st.warning("⚠️ Patient cases database not available. Please upload dataset in Admin Panel.")
    
    # Search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        patient_symptoms = st.text_area(
            "Enter Patient Symptoms:",
            height=150,
            placeholder="e.g., 45-year-old male presenting with chest pain, shortness of breath, and sweating for the past 2 hours..."
        )
    
    with col2:
        st.markdown("### Search Options")
        num_cases = st.slider("Number of similar cases", 1, 10, 5)
        show_details = st.checkbox("Show detailed case information", value=True)
    
    if st.button("🔍 Search Similar Cases", type="primary", use_container_width=True):
        if not patient_symptoms.strip():
            st.warning("Please enter patient symptoms before searching.")
        else:
            with st.spinner("Searching for similar past cases..."):
                similar_cases = search_similar_cases(patient_symptoms, mistral_client, top_k=num_cases)
                
                if similar_cases:
                    st.success(f"Found {len(similar_cases)} similar case(s)!")
                    
                    for idx, case in enumerate(similar_cases, 1):
                        with st.container():
                            st.markdown(f"### 📋 Case {idx} (Similarity: {case['similarity_score']:.2%})")
                            
                            # Display case document - FULL TEXT, no truncation
                            with st.expander("📄 Case Details (Full Text)", expanded=True):
                                # Use text_area for better display of long text
                                st.text_area(
                                    "Complete Patient Case Information:",
                                    value=case['document'],
                                    height=300,
                                    key=f"case_{idx}_full",
                                    label_visibility="collapsed"
                                )
                                # Also show as markdown for better formatting
                                st.markdown(f"**Full Case Text:**")
                                st.markdown(f"```\n{case['document']}\n```")
                            
                            # Display metadata if available
                            if show_details and case['metadata']:
                                with st.expander("🔍 Additional Information"):
                                    metadata = case['metadata']
                                    # Display key fields
                                    key_fields = ['patient_id', 'diagnosis', 'treatment', 'age', 'gender', 'medical_history']
                                    for field in key_fields:
                                        if field in metadata:
                                            st.markdown(f"**{field.replace('_', ' ').title()}:** {metadata[field]}")
                                    
                                    # Show all metadata in a table
                                    if len(metadata) > len(key_fields):
                                        st.markdown("**All Metadata:**")
                                        import pandas as pd
                                        meta_df = pd.DataFrame([metadata])
                                        st.dataframe(meta_df.T, use_container_width=True)
                            
                            st.divider()
                    
                    # AI Analysis of similar cases
                    if len(similar_cases) > 0:
                        st.markdown("---")
                        st.markdown("### 💡 AI Analysis")
                        try:
                            # Use full document text for AI analysis (not truncated)
                            cases_summary = "\n".join([f"Case {i+1}: {case['document']}" for i, case in enumerate(similar_cases[:3])])
                            analysis_prompt = f"""Based on the current patient symptoms: "{patient_symptoms}"

And these similar past cases:
{cases_summary}

Provide a brief clinical analysis comparing the current case with past cases. Highlight:
1. Similarities in symptoms and presentation
2. Potential diagnosis considerations
3. Treatment approaches that worked in similar cases

Keep the response concise and clinically relevant."""
                            
                            analysis_response = mistral_client.chat.complete(
                                model=CHAT_MODEL,
                                messages=[{"role": "user", "content": analysis_prompt}]
                            )
                            
                            st.info(analysis_response.choices[0].message.content)
                        except Exception as e:
                            st.warning(f"Could not generate AI analysis: {str(e)}")
                else:
                    st.error("No similar cases found. The database may be empty or symptoms don't match any cases.")
    
    # Show collection info
    st.markdown("---")
    st.markdown("### 📚 About Clinical Decision Support")
    st.markdown("""
    This feature uses vector similarity search to find past patient cases with similar symptoms.
    It helps clinicians by:
    - Finding similar historical cases
    - Comparing current patient with past cases
    - Providing AI-powered analysis of similarities
    - Aiding in diagnosis and treatment decisions
    """)

elif page == "⚙️ Admin Panel":
    st.title("⚙️ Admin Panel")
    st.markdown("Manage patient dataset and system configuration.")
    
    # Admin tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Dataset", "📊 Collection Stats", "🔗 ChromaDB Info"])
    
    with tab1:
        st.markdown("### 📤 Upload PMC-Patients Dataset")
        st.markdown("""
        Upload the PMC-Patients dataset CSV file to create embeddings for clinical decision support.
        
        **Dataset Source:** [Kaggle - PMC-Patients Dataset](https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support)
        """)
        
        # Option to use file path or upload
        use_file_path = st.checkbox("Use file path instead of upload (for large files)", value=False,
                                    help="If upload fails due to size limits, use this option to process file from your filesystem")
        
        uploaded_file = None
        file_path_input = None
        
        if use_file_path:
            file_path_input = st.text_input(
                "Enter full path to CSV file",
                placeholder="/path/to/PMC-Patients.csv",
                help="Enter the absolute path to your CSV file on your computer"
            )
            if file_path_input and os.path.exists(file_path_input):
                st.success(f"✅ File found: {file_path_input}")
            elif file_path_input:
                st.error(f"❌ File not found: {file_path_input}")
        else:
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload the PMC-Patients.csv file from the Kaggle dataset"
            )
            
            if uploaded_file is not None:
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > 200:
                    st.warning(f"⚠️ **File is {file_size_mb:.1f} MB. If upload fails, check `.streamlit/config.toml` and restart Streamlit.**")
        
        # Determine which file source to use and get file info
        file_ready = False
        file_size_mb = 0
        
        if use_file_path and file_path_input and os.path.exists(file_path_input):
            file_ready = True
            file_size_mb = os.path.getsize(file_path_input) / (1024 * 1024)
        elif uploaded_file is not None:
            file_ready = True
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_ready:
            
            # Show file info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            with col2:
                st.metric("Status", "Ready to Process" if file_size_mb > 0 else "Empty")
            
            # Warning for large files
            if file_size_mb > 200:
                st.warning(f"⚠️ **Large File Detected ({file_size_mb:.1f} MB)**\n\n"
                          f"This file is quite large. Processing may take a significant amount of time and API costs.\n\n"
                          f"**Estimated Processing Time:** {file_size_mb * 0.5:.0f}-{file_size_mb * 2:.0f} minutes\n"
                          f"**Recommendation:** Consider processing a sample first to test.")
            
            # Processing options
            st.markdown("### ⚙️ Processing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                process_sample = st.checkbox("Process Sample First", value=file_size_mb > 200, 
                                            help="Process only first N rows to test")
                if process_sample:
                    sample_size = st.number_input("Sample Size", min_value=100, max_value=10000, 
                                                 value=1000, step=100,
                                                 help="Number of rows to process in sample")
            
            with col2:
                process_full = st.checkbox("Process Full Dataset", value=not (file_size_mb > 200),
                                          help="Process entire dataset (may take hours for large files)")
            
            # Determine file path (will be set when processing starts)
            tmp_path = None
            
            # Process button
            if (process_sample or process_full) and st.button("🚀 Process Dataset", type="primary"):
                # Determine file path
                if use_file_path and file_path_input:
                    tmp_path = file_path_input
                else:
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                
                max_rows = sample_size if process_sample and not process_full else None
                
                if max_rows:
                    st.info(f"📝 Processing sample: First {max_rows:,} rows")
                else:
                    st.info(f"📝 Processing full dataset (this may take a while for {file_size_mb:.1f} MB file)")
                
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    result = process_patient_dataset(tmp_path, mistral_client, progress_bar, status_text, max_rows)
                    
                    status_text.empty()
                    
                    if result['success']:
                        st.success(f"""
                        ✅ **Processing Complete!**
                        - Processed: {result['processed']:,} records
                        - Failed: {result['failed']:,} records
                        - Total in collection: {result['collection_count']:,} cases
                        - File size: {result.get('file_size_mb', 0):.1f} MB
                        """)
                        
                        if max_rows and process_full:
                            st.info("💡 Sample processing complete. Uncheck 'Process Sample First' and check 'Process Full Dataset' to continue with the rest.")
                    else:
                        st.error(f"❌ Error processing dataset: {result.get('error', 'Unknown error')}")
                        if 'Memory' in result.get('error', ''):
                            st.warning("💡 **Memory Error**: Try processing a smaller sample first.")
                
                except Exception as e:
                    status_text.empty()
                    st.error(f"❌ Processing failed: {str(e)}")
                    if 'memory' in str(e).lower() or 'Memory' in str(e):
                        st.warning("💡 **Memory Issue**: The file is too large to process at once. Try processing a sample first.")
                
                finally:
                    # Clean up temp file (only if it was uploaded, not if using file path)
                    if not use_file_path and tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Download the PMC-Patients dataset from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support)
        2. Extract the CSV file (PMC-Patients.csv)
        3. Upload it using the file uploader above
        4. **For large files (>200MB)**: 
           - Start with "Process Sample First" to test (recommended)
           - Then process the full dataset
        5. Click "Process Dataset" to create embeddings
        6. The embeddings will be stored in ChromaDB collection: `patient_cases`
        
        **Note:** Large files (500MB+) may take several hours to process. The system processes data in chunks to manage memory efficiently.
        """)
    
    with tab2:
        st.markdown("### 📊 Collection Statistics")
        
        # Patient collection stats
        patient_stats = get_collection_stats()
        if patient_stats['success']:
            st.metric("Patient Cases", patient_stats['count'])
            st.info(f"Collection: `{patient_stats['collection_name']}`")
        else:
            st.error(f"Error: {patient_stats.get('error', 'Unknown error')}")
        
        # Doctor collection stats
        st.markdown("---")
        st.markdown("### Doctor Collection")
        try:
            doctor_count = collection.count()
            st.metric("Doctor Records", doctor_count)
            st.info(f"Collection: `{COLLECTION_NAME}`")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab3:
        st.markdown("### 🔗 ChromaDB Connection Info")
        st.code(f"""
        Database: {CHROMA_DATABASE}
        Tenant: {CHROMA_TENANT}
        Doctor Collection: {COLLECTION_NAME}
        Patient Collection: patient_cases
        """)
        
        st.markdown("### 🌐 ChromaDB Web Interface")
        st.markdown(f"""
        Access your ChromaDB collections via the web interface:
        - [ChromaDB Console](https://www.trychroma.com/vmjs1412/aws-us-east-1/{CHROMA_DATABASE})
        - Doctor Embeddings: [View Collection](https://www.trychroma.com/vmjs1412/aws-us-east-1/{CHROMA_DATABASE}/collections/{COLLECTION_NAME})
        - Patient Cases: [View Collection](https://www.trychroma.com/vmjs1412/aws-us-east-1/{CHROMA_DATABASE}/collections/patient_cases)
        """)
        
        if st.button("🔄 Refresh Stats"):
            st.rerun()

# Sidebar with info (only show on home page)
if page == "🏠 Home - Doctor Matching":
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
else:
    # Sidebar info for other pages
    with st.sidebar:
        st.markdown("### 📊 System Info")
        st.code(f"""
        Model: {CHAT_MODEL}
        Embedding: {EMBEDDING_MODEL}
        Database: {CHROMA_DATABASE}
        """)
        
        if st.button("🔄 Refresh Connection"):
            st.cache_resource.clear()
            st.rerun()


