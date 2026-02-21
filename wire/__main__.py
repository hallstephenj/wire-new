import os
import uvicorn
from wire.config import load_config

cfg = load_config()
port = int(os.environ.get("PORT", cfg["server"]["port"]))
uvicorn.run("wire.app:app", host=cfg["server"]["host"], port=port, reload=False)
