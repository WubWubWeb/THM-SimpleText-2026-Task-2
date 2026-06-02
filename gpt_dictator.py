import io
import json
import asyncio
from openai import AsyncOpenAI


class LLMCouncil:
    def __init__(self, openai_token_path="openai_token.txt"):
        with open(openai_token_path, "r") as f:
            openai_key = f.read().strip()

        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.model = "gpt-4.1-nano"

    # -------------------------------------------------------------------------
    # REAL-TIME MODE  (single entry, instant results)
    # -------------------------------------------------------------------------

    async def _ask_chatgpt(self, prompt, max_tokens=256):
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ChatGPT Error: {str(e)}"

    async def deliberate(self, prompts_dict, source_text, simplified_text, snt_id):
        tasks, metadata = [], []
        for name, (template, max_tokens) in prompts_dict.items():
            full_prompt = template.format(src=source_text, simp=simplified_text)
            tasks.append(self._ask_chatgpt(full_prompt, max_tokens))
            metadata.append(name)

        responses = await asyncio.gather(*tasks)
        return [
            {"snt_id": snt_id, "task": task_name, "model": "ChatGPT", "response": response}
            for task_name, response in zip(metadata, responses)
        ]

    async def deliberate_batch_realtime(self, prompts_dict, entries, concurrency=10):
        """Real-time path: parallel async calls, results available immediately."""
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

    # -------------------------------------------------------------------------
    # BATCH API MODE  (50 % cheaper, up to 24 h turnaround)
    # -------------------------------------------------------------------------

    def _build_jsonl(self, prompts_dict, entries):
        """
        Build an in-memory JSONL file and a matching index so we can
        reconstruct results after the batch completes.

        Each line is one OpenAI batch request.
        custom_id format:  "<snt_id>||<task_name>"
        """
        lines = []

        for entry in entries:
            snt_id = entry["snt_id"]
            src    = entry["source_sentence"]
            simp   = entry["simplified_sentence"]

            for task_name, (template, max_tokens) in prompts_dict.items():
                full_prompt = template.format(src=src, simp=simp)
                custom_id   = f"{snt_id}||{task_name}"

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": full_prompt}],
                    },
                }
                lines.append(json.dumps(request))

        jsonl_bytes = "\n".join(lines).encode("utf-8")
        return jsonl_bytes

    async def submit_batch(self, prompts_dict, entries):
        """
        Upload the JSONL file and create the batch job.
        Returns the batch object (contains .id for polling).
        """
        jsonl_bytes = self._build_jsonl(prompts_dict, entries)

        # Upload as a file
        file_obj = await self.openai_client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )

        # Create the batch
        batch = await self.openai_client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        print(f"[LLMCouncil] Batch submitted — id: {batch.id}  status: {batch.status}")
        return batch

    async def poll_batch(self, batch_id, poll_interval=30):
        """
        Poll until the batch reaches a terminal state.
        Returns the completed batch object.
        """
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = await self.openai_client.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(
                f"[LLMCouncil] Polling batch {batch_id} — "
                f"status: {batch.status}  "
                f"done: {counts.completed}/{counts.total}  "
                f"failed: {counts.failed}"
            )
            if batch.status in terminal:
                return batch
            await asyncio.sleep(poll_interval)

    async def fetch_batch_results(self, batch, prompts_dict, entries):
        """
        Download output file and parse into the same record format used by
        deliberate() / deliberate_batch_realtime().
        """
        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch.id} finished with status '{batch.status}', "
                "cannot retrieve results."
            )

        # Download raw output
        file_response = await self.openai_client.files.content(batch.output_file_id)
        raw = file_response.text  # newline-delimited JSON

        # Parse each output line into a lookup map
        output_map = {}   # custom_id -> response text
        for line in raw.strip().splitlines():
            obj       = json.loads(line)
            custom_id = obj["custom_id"]
            try:
                text = obj["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                text = f"Error: {json.dumps(obj.get('error', 'unknown'))}"
            output_map[custom_id] = text

        # Reconstruct records in a deterministic order
        results = []
        for entry in entries:
            snt_id = entry["snt_id"]
            for task_name in prompts_dict:
                custom_id = f"{snt_id}||{task_name}"
                results.append({
                    "snt_id":   snt_id,
                    "task":     task_name,
                    "model":    "ChatGPT",
                    "response": output_map.get(custom_id, "Error: missing from output"),
                })
        return results

    async def deliberate_batch(
        self,
        prompts_dict,
        entries,
        *,
        use_batch_api=True,
        poll_interval=30,
        concurrency=10,
    ):
        """
        Main entry point for bulk processing.

        use_batch_api=True  → cheaper (-50%), waits up to 24h
        use_batch_api=False → real-time async, instant but full price
        """
        if use_batch_api:
            batch = await self.submit_batch(prompts_dict, entries)
            batch = await self.poll_batch(batch.id, poll_interval=poll_interval)
            return await self.fetch_batch_results(batch, prompts_dict, entries)
        else:
            return await self.deliberate_batch_realtime(
                prompts_dict, entries, concurrency=concurrency
            )


# ---------------------------------------------------------------------------
# Convenience helpers (mirrors original API)
# ---------------------------------------------------------------------------


async def run_council_batch(prompts, entries, use_batch_api=True, concurrency=10):
    council = LLMCouncil()
    return await council.deliberate_batch(
        prompts,
        entries,
        use_batch_api=use_batch_api,
        concurrency=concurrency,
    )