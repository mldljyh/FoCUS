import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


RUBRICS = {
    "attribute": (
        "You are a strict evaluation judge. You will be given a 'Sentence' (prediction) and a 'Phrase' "
        "(ground truth description). Your task is to rate how accurately and completely the Sentence covers "
        "the detailed attributes (e.g., color, shape, material, design) described in the Phrase. \n"
        "Rate on a scale from 0 to 5:\n"
        "0: Irrelevant or contradictory.\n"
        "1: Mentions the correct object but misses all specific attributes.\n"
        "2: Captures the object and very minor details.\n"
        "3: Captures the object and some key attributes.\n"
        "4: Captures the object and most attributes.\n"
        "5: Perfectly captures the object and all attributes described in the Phrase.\n"
        "Explain the reason and then output the final score within <score></score>."
    ),
    "relation": (
        "You are a strict evaluation judge. You will be given a 'Sentence' (prediction) and a 'Phrase' "
        "(ground truth spatial relationship). Your task is to rate how accurately the Sentence describes "
        "the spatial relationship or interaction defined in the Phrase. \n"
        "Rate on a scale from 0 to 5:\n"
        "0: Relationship is missing, incorrect, or unrelated.\n"
        "1: Mentions the objects but the relationship is wrong.\n"
        "3: Captures the relationship vaguely or partially.\n"
        "4: Correctly describes the relationship, but misses minor nuances or lacks full precision.\n"
        "5: Perfectly captures the spatial relationship or interaction.\n"
        "Explain the reason and then output the final score within <score></score>."
    ),
}


def parse_score(text):
    answer = (text or "").rsplit("</think>", 1)[-1]
    scores = re.findall(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*</score>", answer, re.IGNORECASE)
    if scores:
        value = round(float(scores[-1]))
    else:
        match = re.search(r"[+-]?\d+", answer)
        if match is None:
            raise ValueError("The judge response contains no numeric score")
        value = int(match.group())
    return max(0, min(5, value))


class VLLMJudge:
    def __init__(self, base_url, model, max_workers, timeout, max_retries, generation):
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("FOCUS_JUDGE_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.max_workers = max_workers
        self.generation = generation

    def check_ready(self):
        available = [model.id for model in self.client.models.list().data]
        if self.model not in available:
            raise ValueError(f"Judge model {self.model!r} is missing from {self.client.base_url}models: {available}")

    def score(self, comparisons):
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._score_one, comparisons))

    def _score_one(self, comparison):
        kind, sentence, phrase = comparison
        if not sentence or not phrase:
            return 0
        messages = [
            {"role": "system", "content": RUBRICS[kind]},
            {"role": "user", "content": f"Sentence: {sentence}. Phrase: {phrase}."},
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, **self.generation)
        return parse_score(response.choices[0].message.content)
