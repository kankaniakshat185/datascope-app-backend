import requests

files = {'file': open('PhoneBookDataset.xlsx', 'rb')} # I'll assume they have a CSV or Excel in their dir, actually they have PhoneBookDataset.xlsx
resp = requests.post('http://localhost:8000/shap', files=files)
print(resp.json())
