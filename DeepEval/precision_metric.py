import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# ============================================================
# 1. Create your Gemini LangChain model
# ============================================================

google = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


# ============================================================
# 2. Create a DeepEval wrapper around Gemini
# ============================================================

class GeminiModel(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        response = model.invoke(prompt)

        return response.content

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()

        response = await model.ainvoke(prompt)

        return response.content

    def get_model_name(self):
        return "Gemini 3.1 Flash Lite"


# Create DeepEval-compatible model
deepeval_model = GeminiModel(google)


# ============================================================
# 3. Your RAG example
# ============================================================

actual_output = (
    "We offer a 30-day full refund at no extra cost."
)

expected_output = (
    "You are eligible for a 30 day full refund at no extra cost."
)

retrieval_context = [
    "All customers are eligible for a 30 day full refund at no extra cost."
]


# ============================================================
# 4. Create Contextual Precision metric
# ============================================================

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=deepeval_model,
    include_reason=True
)


# ============================================================
# 5. Create test case
# ============================================================

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)


# ============================================================
# 6. Run evaluation
# ============================================================

evaluate(
    test_cases=[test_case],
    metrics=[metric]
)