"""
Patient Data Processing Module
Processes PMC-Patients dataset and creates embeddings for clinical decision support
"""
import pandas as pd
import os
from typing import List, Dict, Optional
from mistralai import Mistral
import chromadb
from dotenv import load_dotenv
import streamlit as st
import time

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "patient-doctor")
PATIENT_COLLECTION_NAME = os.getenv("PATIENT_COLLECTION_NAME", "patient_cases")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
BATCH_SIZE = 50  # Process embeddings in batches
CHUNK_SIZE = 10000  # Read CSV in chunks for large files
MAX_TEXT_LENGTH = 8000  # Maximum characters per text (Mistral API limit ~8192 tokens, ~8000 chars safe)
MAX_BATCH_TOKENS = 100000  # Maximum total tokens per batch


def get_chroma_client():
    """Get ChromaDB client"""
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )


def get_patient_collection():
    """Get or create patient cases collection"""
    client = get_chroma_client()
    return client.get_or_create_collection(name=PATIENT_COLLECTION_NAME)


def create_patient_text(row: pd.Series) -> str:
    """
    Create a searchable text representation of patient case
    Uses ALL columns with FULL text values for comprehensive embedding
    Truncates to MAX_TEXT_LENGTH to respect Mistral API token limits
    """
    text_parts = []
    
    # Process ALL columns in the row to capture complete patient information
    # This ensures we use the full text from every column for embeddings
    for col in row.index:
        if pd.notna(row[col]) and str(row[col]).strip():
            value = str(row[col]).strip()
            
            # Include ALL values, no matter the length
            # Full text is important for accurate semantic matching
            if value:  # Only skip completely empty values
                # Format: "ColumnName: FullValue"
                text_parts.append(f"{col}: {value}")
    
    # Combine all parts with separator
    result = " | ".join(text_parts)
    
    # Truncate if too long to respect Mistral API token limits
    # Keep most important information (first part usually has key data)
    if len(result) > MAX_TEXT_LENGTH:
        # Truncate but keep a note
        result = result[:MAX_TEXT_LENGTH] + "... [truncated for API limit]"
        st.warning(f"⚠️ Text truncated to {MAX_TEXT_LENGTH} chars (Mistral API limit). Some data may be cut off.")
    
    return result


def create_embeddings_batch(mistral_client: Mistral, texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for a batch of texts
    Handles Mistral API token limits by splitting large batches
    """
    try:
        # Check total length and split if needed
        total_chars = sum(len(text) for text in texts)
        
        # If batch is too large, split into smaller batches
        if total_chars > MAX_BATCH_TOKENS:
            # Split into smaller batches
            batch_results = []
            current_batch = []
            current_batch_size = 0
            
            for text in texts:
                text_len = len(text)
                # If adding this text would exceed limit, process current batch first
                if current_batch_size + text_len > MAX_BATCH_TOKENS and current_batch:
                    # Process current batch
                    response = mistral_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        inputs=current_batch
                    )
                    batch_results.extend([item.embedding for item in response.data])
                    # Start new batch
                    current_batch = [text]
                    current_batch_size = text_len
                else:
                    current_batch.append(text)
                    current_batch_size += text_len
            
            # Process remaining batch
            if current_batch:
                response = mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=current_batch
                )
                batch_results.extend([item.embedding for item in response.data])
            
            return batch_results
        else:
            # Normal batch processing
            response = mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=texts
            )
            return [item.embedding for item in response.data]
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error creating embeddings: {error_msg}")
        
        # If it's a token limit error, try with smaller batch
        if "too many tokens" in error_msg.lower() or "3210" in error_msg:
            st.warning("⚠️ Batch too large, splitting into smaller batches...")
            # Recursively split and process
            if len(texts) > 1:
                mid = len(texts) // 2
                first_half = create_embeddings_batch(mistral_client, texts[:mid])
                second_half = create_embeddings_batch(mistral_client, texts[mid:])
                return first_half + second_half
            else:
                # Single text is too long - truncate it
                st.warning(f"⚠️ Single text too long ({len(texts[0])} chars), truncating...")
                truncated = texts[0][:MAX_TEXT_LENGTH]
                response = mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=[truncated]
                )
                return [item.embedding for item in response.data]
        
        return []


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def estimate_total_rows(file_path: str) -> int:
    """Estimate total rows in CSV without loading entire file"""
    try:
        # Read first chunk to get structure
        chunk = pd.read_csv(file_path, nrows=1000)
        file_size = os.path.getsize(file_path)
        chunk_size = chunk.memory_usage(deep=True).sum()
        estimated_rows = int((file_size / chunk_size) * len(chunk))
        return estimated_rows
    except:
        return 0


def process_patient_dataset(file_path: str, mistral_client: Mistral, progress_bar=None, status_text=None, max_rows: Optional[int] = None) -> Dict:
    """
    Process patient dataset CSV file and create embeddings in ChromaDB
    Uses chunked reading for large files to manage memory efficiently
    
    Args:
        file_path: Path to the CSV file
        mistral_client: Mistral AI client
        progress_bar: Streamlit progress bar (optional)
        status_text: Streamlit status text container (optional)
        max_rows: Maximum number of rows to process (None for all)
    
    Returns:
        Dictionary with processing results
    """
    try:
        # Check file size
        file_size_mb = get_file_size_mb(file_path)
        
        # Estimate total rows
        estimated_rows = estimate_total_rows(file_path)
        
        if status_text:
            status_text.text(f"📊 File size: {file_size_mb:.1f} MB | Estimated rows: {estimated_rows:,}")
        
        # Get patient collection
        collection = get_patient_collection()
        
        # Check if collection already has data
        existing_count = collection.count()
        if existing_count > 0:
            st.warning(f"⚠️ Collection already contains {existing_count:,} records. New records will be added.")
        
        # Process in chunks for large files
        processed = 0
        failed = 0
        batch_texts = []
        batch_indices = []
        batch_metadatas = []
        batch_ids = []
        total_processed = 0
        error_samples = []  # Store sample errors for debugging
        
        # Use chunked reading for large files
        # Try to read with different encodings if needed
        try:
            # First, peek at the CSV structure
            sample_df = pd.read_csv(file_path, nrows=5, low_memory=False)
            columns_found = list(sample_df.columns)
            if status_text:
                status_text.text(f"📋 CSV Columns detected: {len(columns_found)} columns")
            st.info(f"📋 **CSV Structure:** {len(columns_found)} columns found. Sample: {', '.join(columns_found[:10])}")
            
            # Now create the chunk reader
            chunk_reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
        except UnicodeDecodeError:
            # Try with different encoding
            sample_df = pd.read_csv(file_path, nrows=5, encoding='latin-1', low_memory=False)
            columns_found = list(sample_df.columns)
            if status_text:
                status_text.text(f"📋 CSV Columns detected: {len(columns_found)} columns (latin-1 encoding)")
            st.info(f"📋 **CSV Structure:** {len(columns_found)} columns found. Sample: {', '.join(columns_found[:10])}")
            chunk_reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE, encoding='latin-1', low_memory=False)
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to read CSV: {str(e)}",
                'processed': 0,
                'failed': 0
            }
        
        for chunk_num, df_chunk in enumerate(chunk_reader):
            if status_text:
                status_text.text(f"📦 Processing chunk {chunk_num + 1} ({len(df_chunk):,} rows)...")
            
            # Apply max_rows limit if specified
            if max_rows and total_processed >= max_rows:
                break
            
            if max_rows:
                remaining = max_rows - total_processed
                df_chunk = df_chunk.head(remaining)
            
            for idx, row in df_chunk.iterrows():
                try:
                    # Create searchable text
                    patient_text = create_patient_text(row)
                    
                    if not patient_text or len(patient_text.strip()) < 10:
                        failed += 1
                        # Store sample of failed rows for debugging
                        if len(error_samples) < 3:
                            sample_info = {
                                'row_index': idx,
                                'columns': list(row.index)[:5] if len(row.index) > 0 else [],
                                'text_created': patient_text[:100] if patient_text else "EMPTY"
                            }
                            error_samples.append(sample_info)
                        continue
                    
                    # Create unique ID (use chunk offset for global index)
                    global_idx = chunk_num * CHUNK_SIZE + idx
                    patient_id = f"patient_{row.get('Patient_ID', global_idx)}_{global_idx}"
                    
                    # Prepare metadata
                    metadata = {
                        'patient_id': str(row.get('Patient_ID', global_idx)),
                        'row_index': str(global_idx),
                        'source': 'PMC-Patients-Dataset',
                        'chunk': str(chunk_num)
                    }
                    
                    # Add available fields to metadata (limit to avoid huge metadata)
                    for col in df_chunk.columns:
                        if col not in ['Patient_ID'] and pd.notna(row[col]):
                            # Truncate long values for metadata
                            value = str(row[col])
                            if len(value) > 500:
                                value = value[:500] + "..."
                            metadata[col.lower().replace(' ', '_')] = value
                    
                    batch_texts.append(patient_text)
                    batch_indices.append(global_idx)
                    batch_metadatas.append(metadata)
                    batch_ids.append(patient_id)
                    
                    # Process batch when it reaches batch size
                    if len(batch_texts) >= BATCH_SIZE:
                        if status_text:
                            status_text.text(f"🔄 Creating embeddings for batch ({processed + len(batch_texts)}/{estimated_rows if estimated_rows > 0 else '?'})...")
                        
                        embeddings = create_embeddings_batch(mistral_client, batch_texts)
                        
                        if embeddings:
                            # Add to ChromaDB with FULL TEXT documents (no truncation)
                            # The 'documents' field stores the complete patient case text for embeddings
                            # This ensures accurate semantic matching based on all available information
                            collection.add(
                                ids=batch_ids,
                                embeddings=embeddings,
                                documents=batch_texts,  # Full text from all columns
                                metadatas=batch_metadatas  # Metadata may be truncated, but documents are full
                            )
                            processed += len(batch_texts)
                            total_processed += len(batch_texts)
                            
                            if progress_bar and estimated_rows > 0:
                                progress_bar.progress(min(total_processed / estimated_rows, 1.0))
                            
                            if status_text:
                                status_text.text(f"✅ Processed {total_processed:,} records...")
                        
                        # Reset batch
                        batch_texts = []
                        batch_indices = []
                        batch_metadatas = []
                        batch_ids = []
                        
                        # Check max_rows limit
                        if max_rows and total_processed >= max_rows:
                            break
                    
                except Exception as e:
                    failed += 1
                    # Silently skip failed records - continue processing
                    # Only log first few errors for debugging
                    if failed <= 5:
                        error_msg = str(e)
                        if failed <= 2:  # Only show first 2 errors to avoid spam
                            st.warning(f"⚠️ Skipping row {idx}: {error_msg[:100]}")
                    continue
            
            # Check if we've reached max_rows
            if max_rows and total_processed >= max_rows:
                break
        
        # Process remaining batch
        if batch_texts:
            if status_text:
                status_text.text(f"🔄 Processing final batch ({len(batch_texts)} records)...")
            embeddings = create_embeddings_batch(mistral_client, batch_texts)
            if embeddings:
                # Store FULL TEXT documents in ChromaDB (no truncation)
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_texts,  # Complete patient case text
                    metadatas=batch_metadatas
                )
                processed += len(batch_texts)
                total_processed += len(batch_texts)
                if progress_bar:
                    progress_bar.progress(1.0)
        
        final_count = collection.count()
        
        result = {
            'success': True,
            'processed': processed,
            'failed': failed,
            'total': total_processed,
            'collection_count': final_count,
            'file_size_mb': file_size_mb
        }
        
        # Show summary if some records failed (but that's okay)
        if failed > 0:
            success_rate = (processed / (processed + failed) * 100) if (processed + failed) > 0 else 0
            if success_rate >= 90:
                st.info(f"ℹ️ Processed {processed:,} records successfully. {failed:,} records skipped (acceptable).")
            else:
                st.warning(f"⚠️ Processed {processed:,} records. {failed:,} records failed (may need investigation).")
        
        return result
        
    except Exception as e:
        # Even if there's an error, return what was processed
        processed_count = processed if 'processed' in locals() else 0
        failed_count = failed if 'failed' in locals() else 0
        
        # If we processed some records, consider it partial success
        if processed_count > 0:
            st.warning(f"⚠️ Processing stopped with error, but {processed_count:,} records were successfully processed.")
            return {
                'success': True,  # Partial success
                'processed': processed_count,
                'failed': failed_count,
                'total': processed_count,
                'collection_count': get_patient_collection().count() if 'collection' in locals() else 0,
                'file_size_mb': get_file_size_mb(file_path) if os.path.exists(file_path) else 0,
                'error': f"Processing stopped early: {str(e)}"
            }
        else:
            # Complete failure
            return {
                'success': False,
                'error': str(e),
                'processed': 0,
                'failed': failed_count,
                'file_size_mb': get_file_size_mb(file_path) if os.path.exists(file_path) else 0
            }
        
        # Get patient collection
        collection = get_patient_collection()
        
        # Check if collection already has data
        existing_count = collection.count()
        if existing_count > 0:
            st.warning(f"⚠️ Collection already contains {existing_count} records. New records will be added.")
        
        # Process in batches
        processed = 0
        failed = 0
        batch_texts = []
        batch_indices = []
        batch_metadatas = []
        batch_ids = []
        
        for idx, row in df.iterrows():
            try:
                # Create searchable text
                patient_text = create_patient_text(row)
                
                if not patient_text or len(patient_text.strip()) < 10:
                    failed += 1
                    continue
                
                # Create unique ID
                patient_id = f"patient_{row.get('Patient_ID', idx)}_{idx}"
                
                # Prepare metadata
                metadata = {
                    'patient_id': str(row.get('Patient_ID', idx)),
                    'row_index': str(idx),
                    'source': 'PMC-Patients-Dataset'
                }
                
                # Add available fields to metadata
                for col in df.columns:
                    if col not in ['Patient_ID'] and pd.notna(row[col]):
                        # Truncate long values for metadata
                        value = str(row[col])
                        if len(value) > 500:
                            value = value[:500] + "..."
                        metadata[col.lower().replace(' ', '_')] = value
                
                batch_texts.append(patient_text)
                batch_indices.append(idx)
                batch_metadatas.append(metadata)
                batch_ids.append(patient_id)
                
                # Process batch when it reaches batch size
                if len(batch_texts) >= BATCH_SIZE:
                    embeddings = create_embeddings_batch(mistral_client, batch_texts)
                    
                    if embeddings:
                        # Add to ChromaDB
                        collection.add(
                            ids=batch_ids,
                            embeddings=embeddings,
                            documents=batch_texts,
                            metadatas=batch_metadatas
                        )
                        processed += len(batch_texts)
                        
                        if progress_bar:
                            progress_bar.progress(processed / total_rows)
                    
                    # Reset batch
                    batch_texts = []
                    batch_indices = []
                    batch_metadatas = []
                    batch_ids = []
                    
            except Exception as e:
                failed += 1
                st.warning(f"Error processing row {idx}: {str(e)}")
                continue
        
        # Process remaining batch
        if batch_texts:
            embeddings = create_embeddings_batch(mistral_client, batch_texts)
            if embeddings:
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                processed += len(batch_texts)
                if progress_bar:
                    progress_bar.progress(1.0)
        
        return {
            'success': True,
            'processed': processed,
            'failed': failed,
            'total': total_rows,
            'collection_count': collection.count()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'processed': processed if 'processed' in locals() else 0,
            'failed': failed if 'failed' in locals() else 0
        }


def _is_transient_api_error(msg: str) -> bool:
    m = (msg or "").lower()
    transient_markers = [
        "status 503",
        "overflow",
        "upstream connect error",
        "disconnect/reset before headers",
        "timed out",
        "timeout",
        "rate limit",
    ]
    return any(x in m for x in transient_markers)


def search_similar_cases(symptoms: str, mistral_client: Mistral, top_k: int = 5) -> Optional[List[Dict]]:
    """
    Search for similar past patient cases based on symptoms
    
    Args:
        symptoms: Patient symptoms text
        mistral_client: Mistral AI client
        top_k: Number of similar cases to return
    
    Returns:
        List of similar patient cases with metadata
    """
    retries = 3
    backoff_seconds = [0.6, 1.2, 2.0]

    for attempt in range(retries):
        try:
            # Create embedding for symptoms
            embedding_response = mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[symptoms]
            )
            symptoms_embedding = embedding_response.data[0].embedding
            
            # Search in patient collection
            collection = get_patient_collection()
            
            results = collection.query(
                query_embeddings=[symptoms_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            similar_cases = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    case = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]
                    }
                    similar_cases.append(case)
            
            return similar_cases

        except Exception as e:
            msg = str(e)
            is_last_attempt = attempt == (retries - 1)
            if _is_transient_api_error(msg) and not is_last_attempt:
                # Retry short-lived API/gateway issues.
                time.sleep(backoff_seconds[attempt] if attempt < len(backoff_seconds) else 2.0)
                continue
            st.error(f"Error searching similar cases: {msg}")
            return None

    return None


def get_collection_stats() -> Dict:
    """Get statistics about the patient collection"""
    try:
        collection = get_patient_collection()
        count = collection.count()
        return {
            'success': True,
            'count': count,
            'collection_name': PATIENT_COLLECTION_NAME
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
