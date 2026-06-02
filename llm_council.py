import os
import json
import asyncio
from google import genai
from openai import AsyncOpenAI


class LLMCouncil:
    def __init__(
        self,
        google_token_path="google_token.txt",
        openai_token_path="openai_token.txt",
    ):
        with open(google_token_path, "r") as f:
            google_key = f.read().strip()
        with open(openai_token_path, "r") as f:
            openai_key = f.read().strip()

        self.google_client = genai.Client(api_key=google_key)
        self.gemini_model_id = "gemini-2.5-flash"
        self.openai_client = AsyncOpenAI(api_key=openai_key)

    async def _ask_gemini(self, prompt):
        try:
            response = await self.google_client.aio.models.generate_content(
                model=self.gemini_model_id, contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    async def _ask_chatgpt(self, prompt):
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ChatGPT Error: {str(e)}"

    async def judge_the_debate(self, source, simp, model_a_critique, model_b_critique):
        judge_prompt = f"""
        You are the Supreme Judge of a Text Simplification Council.
        SOURCE: "{source}"
        SIMPLIFICATION: "{simp}"
        CRITIQUE A (Gemini): {model_a_critique}
        CRITIQUE B (ChatGPT): {model_b_critique}

        TASK: Evaluate the accuracy of the critiques. Output trust scores (0.0 to 1.0).
        FORMAT:
        TRUST_MODEL_A: [score]
        TRUST_MODEL_B: [score]
        """
        g_task = self._ask_gemini(judge_prompt)
        c_task = self._ask_chatgpt(judge_prompt)
        return await asyncio.gather(g_task, c_task)

    async def deliberate(self, prompts_dict, source_text, simplified_text, snt_id):
        tasks = []
        metadata = []

        for name, template in prompts_dict.items():
            full_prompt = template.format(src=source_text, simp=simplified_text)
            tasks.append(self._ask_gemini(full_prompt))
            tasks.append(self._ask_chatgpt(full_prompt))
            metadata.append((name, "Gemini"))
            metadata.append((name, "ChatGPT"))

        responses = await asyncio.gather(*tasks)

        results = []
        for (task_name, model_name), response in zip(metadata, responses):
            results.append({
                "snt_id": snt_id,
                "task": task_name,
                "model": model_name,
                "response": response,
            })
        return results

    async def deliberate_batch(self, prompts_dict, entries, concurrency=10):
        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(entry):
            async with semaphore:
                return await self.deliberate(
                    prompts_dict,
                    entry["source_sentence"],
                    entry["simplified_sentence"],
                    entry["snt_id"],
                )

        nested = await asyncio.gather(*[process_one(e) for e in entries])
        return [record for sublist in nested for record in sublist]


async def run_council(prompts, source, simp, snt_id):
    council = LLMCouncil()
    return await council.deliberate(prompts, source, simp, snt_id)


async def run_council_batch(prompts, entries, concurrency=10):
    council = LLMCouncil()
    return await council.deliberate_batch(prompts, entries, concurrency)