import pytest
from src.models import GeminiChainOfThought
from src.config import GeminiConfig, PromptConfig

def test_gemini_chain_of_thought_initialization():
    """GeminiChainOfThoughtの初期化テスト"""
    solver = GeminiChainOfThought()
    assert solver is not None

def test_custom_config():
    """カスタム設定のテスト"""
    gemini_config = GeminiConfig(
        temperature=0.5,
        max_output_tokens=1024
    )
    prompt_config = PromptConfig(
        template="カスタムテンプレート: {question}"
    )
    
    solver = GeminiChainOfThought(
        gemini_config=gemini_config,
        prompt_config=prompt_config
    )
    
    assert solver.gemini_config.temperature == 0.5
    assert solver.prompt_config.template == "カスタムテンプレート: {question}"

@pytest.mark.skip(reason="APIキーが必要なため、通常のテストではスキップ")
def test_solve_problem():
    """問題解決のテスト"""
    solver = GeminiChainOfThought()
    question = "2 + 2は？"
    result = solver.solve_problem(question)
    assert result is not None
    assert isinstance(result, str) 