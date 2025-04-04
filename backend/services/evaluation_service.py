from typing import List, Dict, Any
import time
from models.config import ModelConfig
from .rag_service import RAGService
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class EvaluationService:
    def __init__(self):
        self.rag_service = RAGService()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize DeepEval metrics
        self.contextual_precision = ContextualPrecisionMetric()
        self.contextual_recall = ContextualRecallMetric()
        self.contextual_relevancy = ContextualRelevancyMetric()
        self.answer_relevancy = AnswerRelevancyMetric()
        self.faithfulness = FaithfulnessMetric()

        # Initialize LLM-as-judge
        self.judge_llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0
        )
        
        # Define evaluation prompts
        self.quality_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator of LLM responses. Evaluate the following response 
            based on these criteria:
            1. Accuracy (0-10): How accurate is the information?
            2. Completeness (0-10): How complete is the answer?
            3. Clarity (0-10): How clear and well-structured is the response?
            4. Relevance (0-10): How relevant is the response to the query?
            
            Provide scores and brief explanations for each criterion."""),
            ("user", "Query: {query}\nResponse: {response}")
        ])

        self.factuality_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker evaluating LLM responses. Check the following response 
            against the provided context and evaluate:
            1. Factual Accuracy (0-10): How well does the response align with the context?
            2. Hallucination Level (0-10): How much information is fabricated or not supported by context?
            3. Source Attribution (0-10): How well does the response cite or reference the context?
            
            Provide scores and brief explanations for each criterion."""),
            ("user", "Context: {context}\nQuery: {query}\nResponse: {response}")
        ])

    def _calculate_token_cost(self, model: str, num_tokens: int) -> float:
        # OpenAI pricing (as of 2024)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03}
        }
        
        if model not in pricing:
            return 0.0
            
        return (num_tokens * pricing[model]["input"]) / 1000

    async def _evaluate_with_llm_judge(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> Dict[str, Any]:
        # Quality evaluation
        quality_messages = self.quality_prompt.format_messages(
            query=query,
            response=response
        )
        quality_result = await self.judge_llm.ainvoke(quality_messages)
        
        # Factuality evaluation
        context_str = "\n".join(context)
        factuality_messages = self.factuality_prompt.format_messages(
            context=context_str,
            query=query,
            response=response
        )
        factuality_result = await self.judge_llm.ainvoke(factuality_messages)
        
        return {
            "quality_evaluation": quality_result.content,
            "factuality_evaluation": factuality_result.content
        }

    async def evaluate_models(
        self,
        queries: List[str],
        model_configs: List[ModelConfig]
    ) -> Dict[str, Any]:
        results = {
            "model_results": [],
            "comparison_metrics": {
                "average_latency": {},
                "contextual_precision": {},
                "contextual_recall": {},
                "contextual_relevancy": {},
                "answer_relevancy": {},
                "faithfulness": {},
                "llm_judge_quality": {},
                "llm_judge_factuality": {},
                "total_cost": {}
            }
        }

        for config in model_configs:
            model_results = {
                "model": config.llm_model,
                "embedding_model": config.embedding_model,
                "queries": []
            }

            total_latency = 0
            total_tokens = 0
            test_cases = []
            total_quality_score = 0
            total_factuality_score = 0

            for query in queries:
                # Process query
                start_time = time.time()
                response = await self.rag_service.process_query(query, config)
                latency = time.time() - start_time

                # Create test case for DeepEval
                test_case = LLMTestCase(
                    input=query,
                    actual_output=response["response"],
                    expected_output=response.get("expected_output", ""),
                    retrieval_context=response.get("retrieval_context", [])
                )
                test_cases.append(test_case)

                # Calculate metrics
                tokens = len(self.tokenizer.encode(response["response"]))
                cost = self._calculate_token_cost(config.llm_model, tokens)

                # LLM-as-judge evaluation
                llm_evaluation = await self._evaluate_with_llm_judge(
                    query=query,
                    response=response["response"],
                    context=response.get("retrieval_context", [])
                )

                # Store results
                model_results["queries"].append({
                    "query": query,
                    "response": response["response"],
                    "latency": latency,
                    "tokens": tokens,
                    "cost": cost,
                    "llm_evaluation": llm_evaluation
                })

                # Update totals
                total_latency += latency
                total_tokens += tokens

            # Run DeepEval metrics
            evaluation_results = evaluate(
                test_cases=test_cases,
                metrics=[
                    self.contextual_precision,
                    self.contextual_recall,
                    self.contextual_relevancy,
                    self.answer_relevancy,
                    self.faithfulness
                ]
            )

            # Calculate averages
            num_queries = len(queries)
            results["model_results"].append(model_results)
            
            # Store metrics
            results["comparison_metrics"]["average_latency"][config.llm_model] = total_latency / num_queries
            results["comparison_metrics"]["contextual_precision"][config.llm_model] = evaluation_results["contextual_precision"]
            results["comparison_metrics"]["contextual_recall"][config.llm_model] = evaluation_results["contextual_recall"]
            results["comparison_metrics"]["contextual_relevancy"][config.llm_model] = evaluation_results["contextual_relevancy"]
            results["comparison_metrics"]["answer_relevancy"][config.llm_model] = evaluation_results["answer_relevancy"]
            results["comparison_metrics"]["faithfulness"][config.llm_model] = evaluation_results["faithfulness"]
            results["comparison_metrics"]["llm_judge_quality"][config.llm_model] = total_quality_score / num_queries
            results["comparison_metrics"]["llm_judge_factuality"][config.llm_model] = total_factuality_score / num_queries
            results["comparison_metrics"]["total_cost"][config.llm_model] = self._calculate_token_cost(
                config.llm_model,
                total_tokens
            )

        return results 