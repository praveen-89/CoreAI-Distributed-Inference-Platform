"""
Distributed Inference Platform - PyTorch Model Runner

Handles the loading and execution of HuggingFace Transformers models.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("coreai.worker.model")


class ModelRunner:
    """Wrapper for a HuggingFace causal language model."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        logger.info("Loading model '%s' onto device '%s'...", self.model_id, self.device)
        start_time = time.time()
        
        # We use a small causal LM like gpt2 for demonstration.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Ensure we have a pad token for batched generation if needed, though we process sequentially here.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        
        elapsed = time.time() - start_time
        logger.info("Model '%s' loaded in %.2f seconds.", self.model_id, elapsed)

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 100, temperature: float = 1.0) -> dict[str, Any]:
        """Generate a completion for a list of chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            A dictionary conforming to the ChatCompletionResponse `choices` structure.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Naive prompt formatting for models that aren't instruction-tuned.
        # A real system would use apply_chat_template if the tokenizer supports it.
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        logger.debug("Generating completion (max_tokens=%d, temp=%.2f)", max_tokens, temperature)
        
        # We disable gradient calculation for inference
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Slice the output to only get the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        usage = {
            "prompt_tokens": input_length,
            "completion_tokens": len(generated_tokens),
            "total_tokens": input_length + len(generated_tokens),
        }

        return {
            "choice": {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop"
            },
            "usage": usage
        }
