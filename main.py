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
        
        # Ensure we drop completely empty artifact trailing columns 
        df = df.dropna(axis=1, how='all')
        
        # We need a target column for some ML checks
        # Instead of blindly picking the last column which might be an ID or empty, we use a robust heuristic
        target_col = df.columns[-1] 
        for col in reversed(df.columns):
            # Skip artifact columns
            if str(col).startswith("Unnamed:"): continue
            
            # Skip if more than 50% missing
            if (df[col].isnull().sum() / len(df)) > 0.5: continue
            
            nunique = df[col].nunique()
            # Skip if it has 1 value (too static) or is purely an ID (all values unique)
            if nunique < 2 or (nunique == len(df) and len(df) > 10): continue
            
            target_col = col
            break

        results = run_all_checks(df, target_col)
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok"}
