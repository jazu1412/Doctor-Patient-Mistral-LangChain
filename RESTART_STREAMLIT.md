# Fix: File Upload Size Limit

## Issue
Streamlit is showing "File must be 200.0MB or smaller" even though we've configured it for larger files.

## Solution: Restart Streamlit

The `.streamlit/config.toml` file has been created with `maxUploadSize = 1000` (1GB), but Streamlit needs to be restarted to pick up the new configuration.

### Steps to Fix:

1. **Stop the current Streamlit server:**
   - In the terminal where Streamlit is running, press `Ctrl+C`
   - Or close the terminal window

2. **Restart Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Refresh your browser:**
   - Hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux)
   - Or close and reopen the browser tab

4. **Try uploading again:**
   - The 200MB limit should now be gone
   - You should be able to upload your 544MB file

## Alternative: Use File Path (If Upload Still Fails)

If the upload still doesn't work after restarting, you can process the file directly from your filesystem:

1. Place your `PMC-Patients.csv` file in the project directory
2. The system can be modified to accept a file path instead of upload

## Verify Config

Check that `.streamlit/config.toml` exists and contains:
```toml
[server]
maxUploadSize = 1000
maxMessageSize = 1000
```

## Still Having Issues?

If restarting doesn't work:
1. Check that the config file is in `.streamlit/config.toml` (not `.streamlit/config.toml.txt`)
2. Make sure you're running Streamlit from the project root directory
3. Try clearing Streamlit cache: `streamlit cache clear`
