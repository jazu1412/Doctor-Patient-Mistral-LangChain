# Clinical Decision Support System

## Overview

The Clinical Decision Support feature uses the PMC-Patients dataset to provide AI-powered assistance in clinical decision making by finding similar past patient cases based on symptoms.

## Features

1. **Patient Case Search**: Search for similar past cases using vector similarity
2. **AI Analysis**: Get AI-powered comparison of current case with similar past cases
3. **Dataset Management**: Upload and process patient datasets through admin panel
4. **Embedding Storage**: Patient cases stored in ChromaDB for fast similarity search

## Setup Instructions

### 1. Download the Dataset

Download the PMC-Patients dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support
- **File**: `PMC-Patients.csv`

### 2. ChromaDB Collection Setup

**No manual setup required!** The system automatically creates the `patient_cases` collection when you first upload the dataset.

However, you can verify/view the collection in ChromaDB:
- **ChromaDB Console**: https://www.trychroma.com/vmjs1412/aws-us-east-1/patient-doctor
- **Patient Cases Collection**: https://www.trychroma.com/vmjs1412/aws-us-east-1/patient-doctor/collections/patient_cases

### 3. Upload Dataset

1. Open the Streamlit app
2. Navigate to **⚙️ Admin Panel**
3. Go to **📤 Upload Dataset** tab
4. Upload the `PMC-Patients.csv` file
5. Click **🚀 Process Dataset**

The system will:
- Read the CSV file
- Create searchable text from patient data (symptoms, diagnosis, treatment, etc.)
- Generate embeddings using Mistral AI
- Store in ChromaDB collection `patient_cases`

### 4. Processing Details

- **Batch Processing**: Processes 50 records at a time for efficiency
- **Text Creation**: Combines symptoms, diagnosis, treatment, age, gender, medical history
- **Embeddings**: Uses Mistral Embed model (same as doctor matching)
- **Metadata**: Stores all original CSV fields as metadata for detailed retrieval

## Usage

### Clinical Decision Support Page

1. Navigate to **🔬 Clinical Decision Support** page
2. Enter patient symptoms in the text area
3. Adjust number of similar cases to retrieve (1-10)
4. Click **🔍 Search Similar Cases**

### Results Display

For each similar case, you'll see:
- **Similarity Score**: How similar the case is (0-100%)
- **Case Details**: Full case information
- **Additional Information**: Metadata including diagnosis, treatment, age, gender, etc.
- **AI Analysis**: Comparison of current case with similar past cases

### AI Analysis Features

The AI analysis provides:
1. Similarities in symptoms and presentation
2. Potential diagnosis considerations
3. Treatment approaches that worked in similar cases

## Data Structure

### Patient Case Text Format

The system creates searchable text by combining:
```
Patient ID: [ID] | Symptoms: [symptoms] | Diagnosis: [diagnosis] | 
Treatment: [treatment] | Age: [age] | Gender: [gender] | 
Medical History: [history]
```

### ChromaDB Collection Schema

- **Collection Name**: `patient_cases`
- **ID Format**: `patient_{Patient_ID}_{row_index}`
- **Document**: Searchable text representation
- **Embedding**: 1024-dimensional vector (Mistral Embed)
- **Metadata**: All original CSV columns stored as metadata

## Configuration

Add to `.env` file (optional, defaults provided):

```env
PATIENT_COLLECTION_NAME=patient_cases
EMBEDDING_MODEL=mistral-embed
```

## Performance

- **Processing Speed**: ~50 records per batch
- **Search Speed**: Near-instant vector similarity search
- **Scalability**: Can handle thousands of patient cases

## Troubleshooting

### Collection Not Found
- The collection is created automatically on first upload
- Check ChromaDB connection in Admin Panel

### Processing Errors
- Ensure CSV file is properly formatted
- Check that Mistral API key is valid
- Verify ChromaDB credentials

### No Similar Cases Found
- Database may be empty (upload dataset first)
- Try rephrasing symptoms
- Check collection stats in Admin Panel

## Future Enhancements

Potential improvements:
- Real-time case updates
- Case filtering by diagnosis, age, gender
- Treatment outcome tracking
- Statistical analysis of similar cases
- Export case reports
