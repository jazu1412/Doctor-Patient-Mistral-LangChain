# Large File Handling Guide

## Overview

The system has been optimized to handle large CSV files (500MB+) efficiently using chunked processing and memory management.

## Features for Large Files

### 1. Chunked Processing
- Files are read in chunks of 10,000 rows at a time
- Prevents memory overflow for large datasets
- Processes embeddings in batches of 50 records

### 2. Sample Processing
- Test with a sample first (recommended for large files)
- Process 100-10,000 rows to verify everything works
- Then process the full dataset

### 3. Progress Tracking
- Real-time progress bar
- Status updates showing current chunk being processed
- Estimated time remaining

### 4. Streamlit Configuration
- Upload limit increased to 1GB (via `.streamlit/config.toml`)
- Handles files up to 544MB and larger

## Processing a 544MB File

### Recommended Approach

1. **First, Process a Sample:**
   - Upload the file
   - Check "Process Sample First"
   - Set sample size to 1,000-5,000 rows
   - Click "Process Dataset"
   - Verify results are correct

2. **Then Process Full Dataset:**
   - Uncheck "Process Sample First"
   - Check "Process Full Dataset"
   - Click "Process Dataset"
   - Let it run (may take 2-4 hours for 544MB)

### Processing Time Estimates

For a 544MB file with ~100,000-500,000 rows:
- **Sample (1,000 rows)**: ~2-5 minutes
- **Full dataset**: ~2-4 hours (depending on API rate limits)

### Memory Usage

- **Chunked reading**: Only loads 10,000 rows at a time
- **Batch processing**: Processes 50 embeddings per API call
- **Efficient**: Can handle files larger than available RAM

## Troubleshooting

### "File too large" error
- Check `.streamlit/config.toml` exists with `maxUploadSize = 1000`
- Restart Streamlit after creating config file

### Processing stops mid-way
- Check Mistral API rate limits
- Check ChromaDB connection
- Resume by processing remaining rows (system tracks progress)

### Memory errors
- Reduce chunk size in `patient_processor.py` (CHUNK_SIZE)
- Process smaller samples
- Close other applications

### Slow processing
- Normal for large files
- Each embedding API call takes time
- Estimated: 1-2 seconds per 50 records

## Configuration

### Adjust Chunk Size

Edit `patient_processor.py`:
```python
CHUNK_SIZE = 10000  # Increase for faster processing (uses more memory)
BATCH_SIZE = 50     # Embeddings per API call
```

### Streamlit Config

File: `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 1000  # MB
maxMessageSize = 1000
```

## Best Practices

1. **Always test with sample first**
2. **Monitor progress** - don't close browser
3. **Check API limits** - Mistral may have rate limits
4. **Process during off-peak hours** for large files
5. **Keep browser tab open** during processing
6. **Check collection stats** periodically

## Cost Considerations

For a 544MB file:
- **Estimated rows**: ~100,000-500,000
- **API calls**: ~2,000-10,000 embedding requests
- **Cost**: Depends on Mistral pricing (check current rates)

## Resume Processing

If processing stops:
1. Check collection stats in Admin Panel
2. Note how many records were processed
3. You can re-upload and it will add to existing collection
4. ChromaDB handles duplicates (based on ID)

## Performance Tips

1. **Use sample first** to verify setup
2. **Process during low-traffic times** for API
3. **Monitor collection stats** to track progress
4. **Keep connection stable** - don't close browser
5. **Check logs** if errors occur
