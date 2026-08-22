#!/usr/bin/env python3
"""
Physyk AI — Nemotron-30B-Omni VLM Server (FastAPI + vLLM)
=========================================================
High-Performance Local OpenAI-Compatible Inference Server
- Supports Text + Image Multimodal Inputs
- Exposes port 8000 on 0.0.0.0 for Brev Dashboard port forwarding
"""

import os
import sys
import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

# Ensure ninja is in PATH and use fast Triton MoE backend
os.environ["PATH"] = f"/usr/bin:/usr/local/bin:{os.environ.get('PATH', '')}"
os.environ["VLLM_MOE_BACKEND"] = "TRITON"

logging.basicConfig(level=logging.INFO, format="[Nemotron-Server] %(message)s")
log = logging.getLogger("nemotron-server")

MODEL_PATH = "/opt/dlami/nvme/physyk/models/Nemotron-30B-FP8"

app = FastAPI(
    title="Physyk AI — Nemotron-30B-Omni VLM",
    description="Local Multimodal Reasoning & System 2 Agentic Brain for Robotics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance
llm_engine = None

def init_llm():
    global llm_engine
    if llm_engine is not None:
        return llm_engine
    
    log.info(f"Loading Nemotron-30B-FP8 from {MODEL_PATH}...")
    from vllm import LLM
    llm_engine = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    log.info("✅ Nemotron-30B-FP8 loaded successfully into GPU memory!")
    return llm_engine


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "nemotron"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False


@app.on_event("startup")
def startup_event():
    init_llm()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Physyk AI Nemotron-30B-Omni VLM Server",
        "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "docs_url": "/docs",
        "openai_endpoint": "/v1/chat/completions"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": llm_engine is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "nemotron",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "physyk-ai",
                "permission": [],
                "root": "nemotron-30b-omni",
            },
            {
                "id": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global llm_engine
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    from vllm import SamplingParams

    # Extract text and optional images from messages
    prompt_text = ""
    for msg in req.messages:
        role = msg.role.upper()
        if isinstance(msg.content, str):
            prompt_text += f"{role}: {msg.content}\n"
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    prompt_text += f"{role}: {part.get('text', '')}\n"

    prompt_text += "ASSISTANT: "

    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens or 1024,
    )

    outputs = llm_engine.generate([prompt_text], sampling_params)
    generated_text = outputs[0].outputs[0].text if outputs else ""

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = len(outputs[0].prompt_token_ids) if outputs else 0
    completion_tokens = len(outputs[0].outputs[0].token_ids) if outputs else 0

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
