from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os

@dataclass
class GeminiConfig:
    """Geminiモデルの設定オプション"""
    model_name: str = "gemini-2.0-flash-lite"  # モデル名を更新
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 2048
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

class GeminiChainOfThought:
    def __init__(
        self,
        gemini_config: Optional[GeminiConfig] = None,
        prompt_config: Optional[PromptConfig] = None
    ):
        load_dotenv()
        
        self.gemini_config = gemini_config or GeminiConfig()
        self.prompt_config = prompt_config or PromptConfig()
        
        # LLMの初期化
        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_config.model_name,
            temperature=self.gemini_config.temperature,
            top_p=self.gemini_config.top_p,
            top_k=self.gemini_config.top_k,
            max_output_tokens=self.gemini_config.max_output_tokens,
            candidate_count=self.gemini_config.candidate_count,
            stop_sequences=self.gemini_config.stop_sequences,
        )
        
        # プロンプトテンプレートの設定
        self.prompt = PromptTemplate(
            template=self.prompt_config.template,
            input_variables=self.prompt_config.input_variables
        )
        
        # 新しい方式でチェーンを作成
        self.chain = self.prompt | self.llm
    
    def solve_problem(self, question: str) -> str:
        """
        問題を解くメソッド
        """
        # invokeメソッドを使用
        return self.chain.invoke({"question": question})

def main():
    # インスタンス作成
    cot_solver = GeminiChainOfThought()
    
    # 質問の設定
    question = "15個のリンゴが入った箱が3つと、20個のリンゴが入った箱が2つあります。合計で何個のリンゴがありますか？"
    
    # 問題を解いて結果を表示
    result = cot_solver.solve_problem(question)
    print(result)

if __name__ == "__main__":
    main() 