from .models import GeminiChainOfThought
from src.models import GeminiReasoning

def main():
    # インスタンス作成
    cot_solver = GeminiChainOfThought()
    
    # 質問の設定
    question = "15個のリンゴが入った箱が3つと、20個のリンゴが入った箱が2つあります。合計で何個のリンゴがありますか？"
    
    # 問題を解いて結果を表示
    result = cot_solver.solve_problem(question)
    print(result)

def test_reasoning():
    reasoner = GeminiReasoning()
    
    # 直接推論の例
    question = "日本に存在する電柱の本数を教えて"
    direct_result = reasoner.direct_reasoning(question)
    print("直接推論の結果:")
    print(direct_result)
    print("\n" + "="*50 + "\n")
    
    # チェーン推論の例
    chained_result = reasoner.chained_reasoning(question)
    print("チェーン推論の結果:")
    print("\n1. 問題分解:")
    print(chained_result["decomposition"])
    print("\n2. データと仮定:")
    print(chained_result["assumptions"])
    print("\n3. 最終結論:")
    print(chained_result["final_result"])

if __name__ == "__main__":
    main()
    test_reasoning() 