from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
from debugger import run_all_checks

def get_target_column(df: pd.DataFrame) -> str:
    # Use valid clean columns 
    valid_cols = [str(c) for c in df.columns if not str(c).startswith("Unnamed:")]
    if not valid_cols:
        return df.columns[-1]

    # 1. Names indicating a target specifically
    priority_names = ["target", "label", "class", "y", "outcome", "status", "price", "churn", "survived"]
    for col in valid_cols:
        if col.lower() in priority_names: return col
    for col in valid_cols:
        if any(name in col.lower() for name in priority_names): return col

    # 2. Heuristics fallback - Look for a neat binary classification target
    for col in reversed(valid_cols):
        if df[col].nunique() == 2 and not pd.api.types.is_float_dtype(df[col]):
            return col

    # 3. Numeric variables with enough continuous variance 
    for col in reversed(valid_cols):
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            if (df[col].isnull().sum() / len(df)) < 0.2:
                return col

    # 4. Fallback checking ID constraint
    for col in reversed(valid_cols):
        nunique = df[col].nunique()
        if (df[col].isnull().sum() / len(df)) > 0.5: continue
        if nunique >= 2 and not (nunique == len(df) and len(df) > 10):
            return col
            
    return valid_cols[-1]

app = FastAPI(title="Dataset Debugger ML Service")

@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1].lower()
    supported_exts = ['csv', 'xlsx', 'xls', 'json', 'parquet']
    
    if file_ext not in supported_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported: {', '.join(supported_exts)}")
    
    try:
        contents = await file.read()
        
        if file_ext == 'csv':
            df = pd.read_csv(io.BytesIO(contents))
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(contents))
        elif file_ext == 'json':
            df = pd.read_json(io.BytesIO(contents))
        elif file_ext == 'parquet':
            df = pd.read_parquet(io.BytesIO(contents))
        
        # Attempt clean
        df = df.dropna(axis=1, how='all')

        # Advanced target detection algorithm
        target_col = get_target_column(df)

        results = run_all_checks(df, target_col)
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok"}
