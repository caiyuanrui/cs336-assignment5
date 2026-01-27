import asyncio
import json
import os
from pathlib import Path

import aiofiles
from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from cs336_alignment.data import get_MATH_dataset

load_dotenv()

API_KEY = os.environ["DEEPSEEK_API_KEY"]
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-reasoner"
MAX_TOKENS = 4096
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a math question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The thoughts and answer should be concise."
)


class SFTCoordinator:
    def __init__(
        self,
        dataset: Dataset,
        output_path: Path,
        max_concurrent: int = 4,
        max_retries: int = 3,
    ) -> None:
        self.dataset = dataset
        self.output_path = output_path
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

    @staticmethod
    def _get_start_index(output_path: Path):
        if not output_path.exists():
            return 0
        last_line = None
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line

        if last_line is None:
            return 0

        try:
            rec = json.loads(last_line)
            return int(rec["id"]) + 1
        except Exception:
            return 0

    def get_start_index(self) -> int:
        return self._get_start_index(self.output_path)

    async def call_model(
        self, client: AsyncOpenAI, question: str
    ) -> dict[str, str] | None:
        system_prompt = (
            "A conversation between User and Assistant. "
            "The User asks a math question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process in the mind "
            "and then provides the User with the answer. "
            "The thoughts and answer should be concise."
        )

        for attempt in range(self.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    stream=False,
                    max_tokens=MAX_TOKENS,
                )
                reasonging: str = resp.choices[0].message.reasoning_content  # type: ignore
                answer = resp.choices[0].message.content or ""
                return {"reasoning": reasonging, "answer": answer}
            except Exception as e:
                if attempt + 1 >= self.max_retries:
                    print(f"call_model failed after {self.max_retries} retries: {e}")
                    return None
                await asyncio.sleep(2**attempt)

        return None

    async def _worker(
        self,
        client: AsyncOpenAI,
        rxtx_qid: asyncio.Queue[int],  # <-(id, question)
        tx_rec: asyncio.Queue[dict],  # record->
    ):
        while True:
            qid = await rxtx_qid.get()
            if qid < 0 or qid >= len(self.dataset):
                rxtx_qid.task_done()
                return

            row = self.dataset[qid]
            question: str = row["problem"]
            result = await self.call_model(client, question)

            record = {
                "id": qid,
                "question": question,
                "reasoning": "",
                "answer": "",
            }

            if result is not None:
                record["reasoning"] = result["reasoning"]
                record["answer"] = result["answer"]

            await tx_rec.put(record)
            rxtx_qid.task_done()

    async def _dump(self, start_idx: int, rx: asyncio.Queue[dict]):
        next_idx = start_idx
        buffer: dict[int, dict] = {}

        async with aiofiles.open(self.output_path, "a", encoding="utf-8") as writer:
            while next_idx < len(self.dataset):
                record = await rx.get()
                idx = record["id"]
                buffer[idx] = record

                while next_idx in buffer:
                    record = buffer.pop(next_idx)
                    await writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    await writer.flush()
                    next_idx += 1

    async def run(self):
        start_idx = self.get_start_index()
        queue_question_indices: asyncio.Queue[int] = asyncio.Queue(
            maxsize=self.max_concurrent * 2
        )
        queue_records: asyncio.Queue[dict] = asyncio.Queue(
            maxsize=self.max_concurrent * 2
        )

        writer = asyncio.create_task(self._dump(start_idx, queue_records))

        async with (
            AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL) as client,
        ):
            workers = [
                asyncio.create_task(
                    self._worker(client, queue_question_indices, queue_records)
                )
                for _ in range(self.max_concurrent)
            ]

            for i in tqdm(
                range(start_idx, len(self.dataset)),
                initial=start_idx,
                total=len(self.dataset) - start_idx,
            ):
                await asyncio.sleep(1)
                await queue_question_indices.put(i)

            await queue_question_indices.join()

            for _ in workers:
                await queue_question_indices.put(-1)

            for w in workers:
                await w

        await writer


async def main():
    root_dir = Path(__file__).parent.parent.parent
    output_path = root_dir / "data/MATH/sft.jsonl"
    dataset = get_MATH_dataset()
    coord = SFTCoordinator(
        dataset=dataset,
        output_path=output_path,
        max_concurrent=16,
        max_retries=3,
    )
    await coord.run()
    print("[INFO] Done")


if __name__ == "__main__":
    asyncio.run(main())
