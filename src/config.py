from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GeminiConfig:
    """Geminiモデルの設定オプション"""
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
    candidate_count: int = 1
    stop_sequences: Optional[list] = None

@dataclass
class PromptConfig:
    """プロンプトの設定オプション"""
    template: str = "問題を段階的に考えて、最終的な答えを出してください。\n\n問題: {question}\n\n思考プロセス:"
    input_variables: list = None
    
    def __post_init__(self):
        if self.input_variables is None:
            self.input_variables = ["question"] 