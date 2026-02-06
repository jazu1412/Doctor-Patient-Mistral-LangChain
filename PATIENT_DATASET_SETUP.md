# Patient Dataset Setup Guide

## Quick Start

### 1. Download the Dataset

Download the PMC-Patients dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support
- **File to download**: `PMC-Patients.csv`

### 2. ChromaDB Collection

**✅ No manual setup needed!** The system automatically creates the `patient_cases` collection.

However, you can view/manage it via:
- **ChromaDB Web Console**: https://www.trychroma.com/vmjs1412/aws-us-east-1/patient-doctor/collections/patient_cases

### 3. Upload via Admin Panel

1. Run the Streamlit app: `streamlit run app.py`
2. Navigate to **⚙️ Admin Panel** (in sidebar navigation)
3. Go to **📤 Upload Dataset** tab
4. Upload `PMC-Patients.csv`
5. Click **🚀 Process Dataset**

The system will automatically:
- Parse the CSV file
- Create embeddings for each patient case
- Store in ChromaDB collection `patient_cases`

## What Gets Created

### ChromaDB Collection: `patient_cases`

- **Collection Name**: `patient_cases` (configurable via `PATIENT_COLLECTION_NAME` in `.env`)
- **Data Stored**:
  - Patient case text (symptoms, diagnosis, treatment, etc.)
  - Embeddings (1024-dimensional vectors)
  - Metadata (all original CSV columns)

### Searchable Text Format

For each patient, the system creates searchable text combining:
- Patient ID
- Symptoms/Chief Complaint
- Diagnosis
- Treatment
- Age, Gender
- Medical History

## Usage

### Clinical Decision Support

1. Navigate to **🔬 Clinical Decision Support** page
2. Enter patient symptoms
3. Click **🔍 Search Similar Cases**
4. View similar past cases with:
   - Similarity scores
   - Full case details
   - AI-powered analysis

## Configuration

Optional environment variables (in `.env`):

```env
PATIENT_COLLECTION_NAME=patient_cases
EMBEDDING_MODEL=mistral-embed
```

## Troubleshooting

### "Collection not found"
- Collection is created automatically on first upload
- Check ChromaDB connection in Admin Panel

### Processing fails
- Verify CSV file format
- Check Mistral API key
- Ensure ChromaDB credentials are correct

### No similar cases found
- Upload dataset first (Admin Panel)
- Check collection stats in Admin Panel
- Try different symptom descriptions

## Files Created

- `patient_processor.py`: Patient data processing module
- `CLINICAL_DECISION_SUPPORT.md`: Detailed documentation
- Updated `app.py`: Added admin panel and clinical decision support pages
