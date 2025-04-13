from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, List, Optional
from enum import Enum
import streamlit as st
from dotenv import load_dotenv
import os

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキーを取得
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class BaseModel:
    """モデルの基底クラス"""
    def __init__(self):
        # 立場の定義（必要に応じて変更可能）
        self.position_a = {
            "name": "革新的な思考",
            "emoji": "🚀",
            "focus": "新しいアイデアや斬新なアプローチを重視",
            "style": "info"
        }
        self.position_b = {
            "name": "保守的な思考",
            "emoji": "🛡️",
            "focus": "実現可能性やリスク、既存の枠組みを重視",
            "style": "warning"
        }
        self.llm = self._initialize_llm()
        self.chain = self._create_chain()
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """LLMを初期化"""
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GOOGLE_API_KEY,  # APIキーを指定
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def _create_chain(self) -> LLMChain:
        """チェーンを作成"""
        raise NotImplementedError
    
    def _get_llm_response(self, prompt: str) -> str:
        """LLMの応答を取得"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

class GeminiChainOfThought:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.chain = self._initialize_chain()
    
    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def _initialize_chain(self):
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question"],
                template="""
                以下の質問について、段階的に考えて回答してください。

                質問: {question}

                1. まず、問題を理解し、必要な情報を整理します。
                2. 次に、段階的に推論を進めていきます。
                3. 最後に、結論を導き出します。

                各ステップの思考プロセスを明確に示してください。
                """
            )
        )
    
    def solve_problem(self, question: str) -> Dict[str, str]:
        """Chain of Thoughtパターンで問題を解決"""
        # 問題分析
        analysis_result = self.chain.invoke({"question": question})
        
        # 思考プロセス構築
        thought_process = self.chain.invoke({
            "question": f"以下の問題分析に基づいて、思考プロセスを構築してください。\n\n{analysis_result['text']}"
        })
        
        # 段階的推論
        reasoning_result = self.chain.invoke({
            "question": f"以下の思考プロセスに基づいて、段階的に推論してください。\n\n{thought_process['text']}"
        })
        
        # 最終回答生成
        final_answer = self.chain.invoke({
            "question": f"以下の推論結果に基づいて、最終的な回答を生成してください。\n\n{reasoning_result['text']}"
        })
        
        return {
            "analysis": analysis_result['text'],
            "thought_process": thought_process['text'],
            "reasoning": reasoning_result['text'],
            "final_answer": final_answer['text']
        }

class GeminiReasoning:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.chain = self._initialize_chain()
    
    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def _initialize_chain(self):
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question"],
                template="""
                以下の質問について、構造化された推論を行ってください。

                質問: {question}

                1. 前提条件を明確にします。
                2. 利用可能なデータを整理します。
                3. 論理的な推論を行います。
                4. 結論を導き出します。

                各ステップの結果を明確に示してください。
                """
            )
        )
    
    def direct_reasoning(self, question: str) -> Dict[str, str]:
        """直接推論パターン"""
        # 前提条件分析
        assumptions = self.chain.invoke({
            "question": f"以下の質問について、前提条件を分析してください。\n\n{question}"
        })
        
        # データ処理
        data_processing = self.chain.invoke({
            "question": f"以下の前提条件に基づいて、データを処理してください。\n\n{assumptions['text']}"
        })
        
        # 推論実行
        reasoning = self.chain.invoke({
            "question": f"以下のデータに基づいて、推論を実行してください。\n\n{data_processing['text']}"
        })
        
        return {
            "assumptions": assumptions['text'],
            "data_processing": data_processing['text'],
            "reasoning": reasoning['text']
        }
    
    def chained_reasoning(self, question: str) -> Dict[str, str]:
        """チェーン推論パターン"""
        # 問題分解
        decomposition = self.chain.invoke({
            "question": f"以下の問題を分解してください。\n\n{question}"
        })
        
        # データ分析
        data_analysis = self.chain.invoke({
            "question": f"以下の問題分解に基づいて、データを分析してください。\n\n{decomposition['text']}"
        })
        
        # 仮定設定
        assumptions = self.chain.invoke({
            "question": f"以下のデータ分析に基づいて、仮定を設定してください。\n\n{data_analysis['text']}"
        })
        
        # 推論実行
        final_result = self.chain.invoke({
            "question": f"以下の仮定に基づいて、最終的な結論を導き出してください。\n\n{assumptions['text']}"
        })
        
        return {
            "decomposition": decomposition['text'],
            "data_analysis": data_analysis['text'],
            "assumptions": assumptions['text'],
            "final_result": final_result['text']
        }

class EvaluatorOptimizer:
    def __init__(self):
        self.generator = self._initialize_llm("generator")
        self.evaluator = self._initialize_llm("evaluator")
        self.optimizer = self._initialize_llm("optimizer")
    
    def _initialize_llm(self, role: str):
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def generate_optimized_response(self, question: str) -> Dict[str, Any]:
        """Evaluator-Optimizerワークフロー"""
        iterations = []
        
        # 初期回答生成
        initial_response = self.generator.invoke({
            "question": question
        })
        iterations.append({
            "iteration": 1,
            "response": initial_response['text']
        })
        
        # 評価
        evaluation = self.evaluator.invoke({
            "question": f"以下の回答を評価し、改善点を指摘してください。\n\n{initial_response['text']}"
        })
        iterations[0]["evaluation"] = evaluation['text']
        
        # 最適化
        optimized_response = self.optimizer.invoke({
            "question": f"以下の評価に基づいて、回答を最適化してください。\n\n{evaluation['text']}"
        })
        iterations[0]["optimized_response"] = optimized_response['text']
        
        return {
            "iterations": iterations,
            "final_response": optimized_response['text']
        }

class DebateBasedCooperation(BaseModel):
    def _create_chain(self) -> LLMChain:
        """ディベートのプロンプトチェーンを作成"""
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question", "position_a_name", "position_b_name"],
                template="""
                以下の質問について、{position_a_name}と{position_b_name}の両方の視点から議論を行い、最適な回答を導き出してください。
                質問: {question}
                """
            )
        )

    def generate_debate_response(self, question: str) -> Dict[str, Any]:
        """ディベートベースの協調パターン"""
        try:
            # 立場Aからの意見
            position_a_opinion = str(self._get_llm_response(
                f"{self.position_a['name']}の視点から以下の質問について意見を述べてください。{self.position_a['focus']}してください。\n質問: {question}"
            ))
            
            # 立場Bからの反論
            position_b_rebuttal = str(self._get_llm_response(
                f"{self.position_b['name']}の視点から以下の意見に対して反論してください。{self.position_b['focus']}してください。\n意見: {position_a_opinion}"
            ))
            
            # 立場Aからの再反論
            position_a_rebuttal = str(self._get_llm_response(
                f"{self.position_a['name']}の視点から以下の反論に対して再反論してください。{self.position_a['focus']}してください。\n反論: {position_b_rebuttal}"
            ))
            
            # 立場Bからの再反論
            position_b_final_rebuttal = str(self._get_llm_response(
                f"{self.position_b['name']}の視点から以下の再反論に対して最終的な反論をしてください。{self.position_b['focus']}してください。\n再反論: {position_a_rebuttal}"
            ))
            
            # 合意形成
            consensus = str(self._get_llm_response(
                f"以下の議論を踏まえて、{self.position_a['name']}と{self.position_b['name']}の両方を考慮した合意形成を行ってください。\n\n{self.position_a['name']}の意見: {position_a_opinion}\n{self.position_b['name']}の反論: {position_b_rebuttal}\n{self.position_a['name']}の再反論: {position_a_rebuttal}\n{self.position_b['name']}の最終反論: {position_b_final_rebuttal}"
            ))
            
            return {
                "iterations": [{
                    "iteration": 1,
                    "position_a_opinion": position_a_opinion,
                    "position_b_rebuttal": position_b_rebuttal,
                    "position_a_rebuttal": position_a_rebuttal,
                    "position_b_final_rebuttal": position_b_final_rebuttal,
                    "position_a": self.position_a,
                    "position_b": self.position_b
                }],
                "final_response": consensus
            }
        except Exception as e:
            st.error(f"Error in Debate-based Cooperation: {str(e)}")
            return {"final_response": "処理中にエラーが発生しました。"} 