import streamlit as st
from models import GeminiChainOfThought, GeminiReasoning, EvaluatorOptimizer
from typing import Dict, Any, List
from enum import Enum

class StepStatus(Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class StepProgress:
    def __init__(self, steps: List[str]):
        self.steps = steps
        self.status = {step: StepStatus.WAITING for step in steps}
        self.current_step = 0
    
    def update_status(self, step: str, status: StepStatus) -> None:
        if step in self.status:
            self.status[step] = status
            if status == StepStatus.PROCESSING:
                self.current_step = self.steps.index(step)
    
    def get_current_step(self) -> int:
        return self.current_step

def display_progress(step_progress: StepProgress, status_container) -> None:
    """進捗状況を表示"""
    if not step_progress:
        return
    
    # ステップの状態を1行で表示
    status_items = []
    current_step_detail = ""
    
    for step, status in step_progress.status.items():
        # ステップの状態に応じたスタイルを設定
        if status == StepStatus.PROCESSING:
            status_items.append(f'<div class="status-item processing">🔄 {step}</div>')
        elif status == StepStatus.COMPLETED:
            status_items.append(f'<div class="status-item completed">✅ {step}</div>')
        elif status == StepStatus.FAILED:
            status_items.append(f'<div class="status-item failed">❌ {step}</div>')
        else:
            status_items.append(f'<div class="status-item">⏳ {step}</div>')
    
    # ステータスを表示
    status_container.markdown(
        f"""
        <style>
            .status-container {{
                width: 100%;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                background: #262730;
                border-radius: 8px;
                margin: 10px 0;
                box-sizing: border-box;
            }}
            .status-item {{
                flex: 1;
                text-align: center;
                margin: 0 5px;
                min-width: 150px;
                padding: 10px 15px;
                border-radius: 8px;
                background: #1E1E1E;
                color: #CCCCCC;
                font-weight: bold;
                border: 1px solid #333333;
            }}
            .status-item.processing {{
                background: #2D2D2D;
                color: #FFA500;
                border: 1px solid #FFA500;
            }}
            .status-item.completed {{
                background: #1E3A1E;
                color: #4CAF50;
                border: 1px solid #4CAF50;
            }}
            .status-item.failed {{
                background: #3A1E1E;
                color: #FF4444;
                border: 1px solid #FF4444;
            }}
        </style>
        {current_step_detail if current_step_detail else ""}
        <div class='status-container'>
            {"".join(status_items)}
        </div>
        """,
        unsafe_allow_html=True
    )

class AIPatternDemo:
    def __init__(self):
        self.cot_solver = GeminiChainOfThought()
        self.reasoner = GeminiReasoning()
        self.evaluator_optimizer = EvaluatorOptimizer()
        
    def direct_query(self, question: str) -> str:
        """単純な質問応答"""
        result = self.reasoner.llm.invoke(question)
        return result
    
    def chain_of_thought(self, question: str) -> Dict[str, str]:
        """Chain of Thoughtパターン"""
        result = self.cot_solver.solve_problem(question)
        return {
            "analysis": result["analysis"],
            "thought_process": result["thought_process"],
            "reasoning": result["reasoning"],
            "final_answer": result["final_answer"]
        }
    
    def direct_reasoning(self, question: str) -> Dict[str, str]:
        """直接推論パターン"""
        result = self.reasoner.direct_reasoning(question)
        return {
            "assumptions": result["assumptions"],
            "data_processing": result["data_processing"],
            "reasoning": result["reasoning"]
        }
    
    def chained_reasoning(self, question: str) -> Dict[str, str]:
        """チェーン推論パターン"""
        result = self.reasoner.chained_reasoning(question)
        return {
            "decomposition": result["decomposition"],
            "data_analysis": result["data_analysis"],
            "assumptions": result["assumptions"],
            "final_result": result["final_result"]
        }
    
    def evaluator_optimizer_workflow(self, question: str) -> Dict[str, Any]:
        """Evaluator-Optimizerワークフロー"""
        result = self.evaluator_optimizer.generate_optimized_response(question)
        return {
            "iterations": [{
                "iteration": iteration["iteration"],
                "response": iteration["response"],
                "evaluation": iteration["evaluation"],
                "optimized_response": iteration["optimized_response"]
            } for iteration in result["iterations"]],
            "final_response": result["final_response"]
        }

def main():
    """メイン関数"""
    st.set_page_config(
        page_title="AIエージェント デザインパターン",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("AIエージェント デザインパターン")
    st.write("異なるAIエージェントのデザインパターンを比較・検証できます")
    
    # パターンの説明を定義
    pattern_descriptions = {
        "シンプルな質問応答": {
            "description": """
            **特徴:**
            - 最もシンプルで直感的なパターン
            - 人間が質問するように、そのままAIに質問を投げかける
            - 複雑な推論や長い回答には不向き
            - 回答の信頼性は比較的低い

            **適しているユースケース:**
            - 単純な事実確認や定義の説明
              - 例：「Pythonとは何ですか？」「東京の人口は？」
            - 簡単な計算問題
              - 例：「2+2は？」「100円の20%引きは？」
            - 短い回答で十分な場合
              - 例：「今日の天気は？」「この単語の意味は？」
            - 素早い回答が必要な場合
              - 例：「現在時刻は？」「次の電車は何時？」

            **例:**
            - 「Pythonとは何ですか？」
            - 「2+2は？」
            - 「東京の人口は？」
            """,
            "example": "Pythonとは何ですか？",
            "steps": ["問題分析", "回答生成"]
        },
        "段階的思考（Chain of Thought）": {
            "description": """
            **特徴:**
            - 人間の思考プロセスを模倣したパターン
            - 「なぜそうなるのか」を段階的に説明
            - 複雑な問題でも正確な回答が得られる
            - 回答の信頼性が高い

            **適しているユースケース:**
            - 数学的な問題解決
              - 例：「15個のリンゴが入った箱が3つと、20個のリンゴが入った箱が2つあります。合計で何個のリンゴがありますか？」
            - 論理的な推論が必要な問題
              - 例：「AさんはBさんより2歳年上で、BさんはCさんより3歳年上です。AさんはCさんより何歳年上ですか？」
            - 複数のステップを要する問題
              - 例：「この数学の問題を解くには、どのような手順で進めればよいですか？」
            - 思考プロセスの説明が重要な場合
              - 例：「なぜこの結論に至ったのか、その理由を説明してください」

            **例:**
            - 「15個のリンゴが入った箱が3つと、20個のリンゴが入った箱が2つあります。合計で何個のリンゴがありますか？」
            - 「AさんはBさんより2歳年上で、BさんはCさんより3歳年上です。AさんはCさんより何歳年上ですか？」
            """,
            "example": "15個のリンゴが入った箱が3つと、20個のリンゴが入った箱が2つあります。合計で何個のリンゴがありますか？",
            "steps": ["問題分析", "思考プロセス構築", "段階的推論", "回答生成"]
        },
        "構造化推論": {
            "description": """
            **特徴:**
            - 問題を構造化して分析するパターン
            - 前提条件、データ、プロセス、結論を明確に分けて考える
            - 論理的な分析に適している
            - 分析結果の再現性が高い

            **適しているユースケース:**
            - 統計データの分析
              - 例：「このデータから何が言えるか、前提条件と分析プロセスを明確にして説明してください」
            - 市場調査の結果解釈
              - 例：「このアンケート結果から、どのような市場動向が読み取れるか分析してください」
            - 技術的な問題の診断
              - 例：「このエラーメッセージの原因を、発生条件と解決手順を明確にして説明してください」
            - 構造化された情報の処理
              - 例：「このレポートの要点を、前提条件と結論を明確にして要約してください」

            **例:**
            - 「日本の少子高齢化の影響を分析してください」
            - 「このエラーメッセージの原因を特定してください」
            """,
            "example": "日本の少子高齢化の影響を分析してください",
            "steps": ["前提条件分析", "データ処理", "推論実行"]
        },
        "連鎖推論": {
            "description": """
            **特徴:**
            - 複数の推論を連鎖させて問題を解決するパターン
            - 1つの推論の結果が次の推論の前提条件になる
            - 複雑な問題を段階的に解決できる
            - 推論の過程が透明で理解しやすい

            **適しているユースケース:**
            - 複雑な意思決定プロセス
              - 例：「新しいビジネスを始める際のリスク評価を、各要因の関連性を考慮して分析してください」
            - 複数の要因が絡む問題
              - 例：「都市計画における交通渋滞の解決策を、経済的影響と環境への影響を考慮して提案してください」
            - 段階的な分析が必要な問題
              - 例：「このプロジェクトの成功要因を、各段階の依存関係を考慮して分析してください」
            - 不確実性の高い問題
              - 例：「将来の市場動向を、様々なシナリオを考慮して予測してください」

            **例:**
            - 「新しいビジネスを始める際のリスク評価」
            - 「都市計画における交通渋滞の解決策」
            """,
            "example": "新しいビジネスを始める際のリスク評価",
            "steps": ["問題分解", "データ分析", "仮定設定", "推論実行"]
        },
        "生成と評価の繰り返し": {
            "description": """
            **特徴:**
            - 生成と評価を繰り返して回答を改善するパターン
            - 2つのAIが協力して高品質な回答を作成
            - 生成AIが回答を作成し、評価AIが改善点を指摘
            - より洗練された回答が得られる

            **適しているユースケース:**
            - クリエイティブなコンテンツ生成
              - 例：「新しい製品のマーケティング戦略を、ターゲット層と競合分析を考慮して提案してください」
            - 技術的なドキュメント作成
              - 例：「このAPIの仕様書を、技術的な正確性と読みやすさを考慮して作成してください」
            - 複雑な問題解決
              - 例：「このビジネスケースの分析を、複数の観点から評価して改善案を提案してください」
            - 高品質な回答が求められる場合
              - 例：「この研究論文の要約を、正確性と簡潔さを考慮して作成してください」

            **例:**
            - 「新しい製品のマーケティング戦略を提案してください」
            - 「技術的なホワイトペーパーの作成」
            - 「複雑なビジネスケースの分析」
            """,
            "example": "新しい製品のマーケティング戦略を提案してください",
            "steps": ["初期回答生成", "評価", "最適化"]
        }
    }
    
    # サイドバーでパターンを選択
    pattern = st.sidebar.selectbox(
        "デザインパターンを選択",
        list(pattern_descriptions.keys())
    )
    
    # 選択されたパターンの説明を表示
    st.sidebar.markdown("### 選択されたパターンの説明")
    st.sidebar.markdown(pattern_descriptions[pattern]["description"])
    
    # 入力エリア（選択されたパターンの例を初期値として設定）
    st.markdown("## 入力フォーム")
    question = st.text_area(
        "質問を入力してください",
        value=pattern_descriptions[pattern]["example"],
        height=100
    )

    # パターン選択時にステータス表示を初期化
    if pattern:
        step_progress = StepProgress(pattern_descriptions[pattern]["steps"])
        # 初期状態を表示
        st.markdown("### 実行中の処理ステップ")
        status_container = st.empty()
        display_progress(step_progress, status_container)
    
    # 実行ボタン
    if st.button("実行"):
        demo = AIPatternDemo()
        
        with st.spinner("AIが考えています..."):
            try:
                if pattern == "シンプルな質問応答":
                    step_progress.update_status("問題分析", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    result = demo.direct_query(question)
                    step_progress.update_status("問題分析", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    step_progress.update_status("回答生成", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    step_progress.update_status("回答生成", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    st.write("### 最終回答")
                    st.write(result)
                
                elif pattern == "段階的思考（Chain of Thought）":
                    step_progress.update_status("問題分析", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    step_progress.update_status("問題分析", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)

                    step_progress.update_status("思考プロセス構築", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    step_progress.update_status("思考プロセス構築", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)

                    step_progress.update_status("段階的推論", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    step_progress.update_status("段階的推論", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)

                    step_progress.update_status("回答生成", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    result = demo.chain_of_thought(question)
                    step_progress.update_status("回答生成", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    st.write("### 最終回答")
                    st.write(result)
                
                elif pattern == "構造化推論":
                    step_progress.update_status("前提条件分析", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    step_progress.update_status("前提条件分析", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)

                    step_progress.update_status("データ処理", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    step_progress.update_status("データ処理", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)

                    step_progress.update_status("推論実行", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    result = demo.direct_reasoning(question)
                    step_progress.update_status("推論実行", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    st.write("### 最終回答")
                    st.write(result)
                
                elif pattern == "連鎖推論":
                    step_progress.update_status("問題分解", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    decomposition = demo.chained_reasoning(question)["decomposition"]
                    step_progress.update_status("問題分解", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    step_progress.update_status("データ分析", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    data_analysis = demo.chained_reasoning(question)["data_analysis"]
                    step_progress.update_status("データ分析", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    step_progress.update_status("仮定設定", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    assumptions = demo.chained_reasoning(question)["assumptions"]
                    step_progress.update_status("仮定設定", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    step_progress.update_status("推論実行", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    final_result = demo.chained_reasoning(question)["final_result"]
                    step_progress.update_status("推論実行", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    st.write("### 4. 最終結論")
                    st.success(final_result)
                
                elif pattern == "生成と評価の繰り返し":
                    step_progress.update_status("初期回答生成", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    initial_response = demo.evaluator_optimizer_workflow(question)["response"]
                    step_progress.update_status("初期回答生成", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    step_progress.update_status("評価", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    evaluation = demo.evaluator_optimizer_workflow(question)["evaluation"]
                    step_progress.update_status("評価", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    step_progress.update_status("最適化", StepStatus.PROCESSING)
                    display_progress(step_progress, status_container)
                    optimized_response = demo.evaluator_optimizer_workflow(question)["optimized_response"]
                    step_progress.update_status("最適化", StepStatus.COMPLETED)
                    display_progress(step_progress, status_container)
                    
                    st.write("### 最終回答")
                    st.success(optimized_response)
            
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                # エラーが発生したステップを失敗としてマーク
                for step in step_progress.steps:
                    if step_progress.status[step] == StepStatus.PROCESSING:
                        step_progress.update_status(step, StepStatus.FAILED)
                display_progress(step_progress, status_container)

if __name__ == "__main__":
    main() 