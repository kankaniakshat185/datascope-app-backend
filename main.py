from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
from debugger import run_all_checks

app = FastAPI(title="Dataset Debugger ML Service")

@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # We need a target column for some checks (classification/regression)
        # For simplicity, we assume the last column is the target
        target_col = df.columns[-1]

        results = run_all_checks(df, target_col)
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok"}
