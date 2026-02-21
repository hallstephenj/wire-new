import uvicorn
from wire.config import load_config

cfg = load_config()
uvicorn.run("wire.app:app", host=cfg["server"]["host"], port=cfg["server"]["port"], reload=False)
