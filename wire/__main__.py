import os
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
from wire.config import load_config

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

cfg = load_config()
port = int(os.environ.get("PORT", cfg["server"]["port"]))
uvicorn.run("wire.app:app", host=cfg["server"]["host"], port=port, reload=False)
