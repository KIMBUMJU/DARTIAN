from fastapi import FastAPI, HTTPException
from app.graph import graph
from pydantic import BaseModel

import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"messages": "Welcome to DARTIAN!"}

@app.post("/dartian")
def dartian(request: QuestionRequest):
    try:
        result = graph.invoke({"question": request.question})

        return {
            "context": result["context"],
            "answer": result["answer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))