"""
fix_cors.py -- run this once to add CORS support to api/main.py so
demo.html (opened as a local file) can call the API without the browser
blocking the request.

Run with:
    python fix_cors.py
"""
from pathlib import Path

main_py = Path("api/main.py")
content = main_py.read_text()

marker = 'app = FastAPI(title="Constrained PPE Detection & Reasoning API")'

if "CORSMiddleware" in content:
    print("CORS already added -- nothing to do.")
else:
    insertion = marker + '''

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)'''
    new_content = content.replace(marker, insertion)
    if new_content == content:
        print("ERROR: could not find the expected line in api/main.py -- no changes made.")
    else:
        main_py.write_text(new_content)
        print("CORS middleware added successfully to api/main.py")
