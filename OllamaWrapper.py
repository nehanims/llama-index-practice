from typing import Optional, List, Mapping, Any
import requests
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from pydantic import Field

import ollama

class OllamaLLM(CustomLLM):
    base_url: str = Field(default="http://192.168.50.35:11434/api/generate")
    modelname: str = Field(..., description="Name of the Ollama model to use")
    context_window: int = Field(default=4096, description="Context window size")
    num_output: int = Field(default=256, description="Number of output tokens")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.modelname,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = requests.post(
            self.base_url,
            json={
                "model": self.modelname,
                "prompt": prompt,
                "stream": False
            },
        )
        result = response.json()
        return CompletionResponse(text=result["response"])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        stream = ollama.chat(#TODO how does it know the base_url???!!
            model=self.modelname,
            messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            delta = chunk['message']['content']
            full_response += delta
            yield CompletionResponse(text=full_response, delta=delta)

# Example usage:
if __name__ == "__main__":
    ollama_llm = OllamaLLM(modelname="llama3.1")

    # Non-streaming example
    response = ollama_llm.complete("Tell me a joke.")
    print(response.text)

    # Streaming example
    for chunk in ollama_llm.stream_complete("Tell me a story."):
        print(chunk.delta, end='', flush=True)
    print()  # New line after streaming is complete


