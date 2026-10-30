"""FastAPI wrapper. Run:  pip install fastapi uvicorn  &&  uvicorn app:app --reload
POST /ask  {"question": "..."}   -> {"answer": "...", "handler": null|"..."}
Any payload (missing field, null, number, 1 MB text) gets a helpful JSON reply, never a stack trace."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from dataclasses import replace
from handlers import MESSAGES
from rag_pipeline import FIXED
from pipeline_perf import Pipeline
from cache import SemanticCache, RedisStore, InMemoryStore

app = FastAPI(title="PayEasy Support Assistant")
MODE = "openai" if os.getenv("OPENAI_API_KEY") else "offline"
_store = RedisStore(url=os.environ["REDIS_URL"]) if os.getenv("REDIS_URL") else InMemoryStore()
pipe = Pipeline(replace(FIXED, min_rel_score=0.8), SemanticCache(_store), mode=MODE)   # optimised config from Day 52

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        question = body.get("question") if isinstance(body, dict) else body
    except Exception:
        return JSONResponse({"answer": MESSAGES["unsafe_input"], "handler": "malformed_request"}, status_code=200)
    out = pipe.run(question)
    return {"answer": out["answer"], "handler": out.get("handler"), "sources": out.get("retrieved_ids", []),
            "cache_hit": out["cache_hit"], "cost_usd": out["cost_usd"], "latency_ms": round(out["latency_ms"])}

@app.exception_handler(Exception)
async def any_error(_, exc):          # last resort: no technical details to the user
    return JSONResponse({"answer": MESSAGES["internal_error"].format(ref="API"), "handler": "internal_error"}, status_code=200)

@app.get("/health")
def health(): return {"status": "ok"}
