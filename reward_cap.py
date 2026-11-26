import re

import spacy
import torch
from sentence_transformers import SentenceTransformer


def judge_comparisons(objects, evidence, sentences):
    def caption(indices):
        return ". ".join(sentences[i] for i in sorted(indices))

    attributes, relations = [], []
    for obj in objects:
        ids = evidence[obj["category"]]
        attributes.append(("attribute", caption(ids), obj["description"]))
        for relation in obj["relations"]:
            related_ids = ids.copy()
            for category, other_ids in evidence.items():
                if category in relation:
                    related_ids.update(other_ids)
            relations.append(("relation", caption(related_ids), relation))
    return attributes, relations


class CaptionReward:
    def __init__(self, judge, nlp, encoder, weights, similarity_threshold, soft_coverage, object_scale):
        self.judge = judge
        self.nlp = nlp
        self.encoder = encoder
        self.weights = weights
        self.similarity_threshold = similarity_threshold
        self.soft_coverage = soft_coverage
        self.object_scale = object_scale

    @classmethod
    def from_config(cls, judge, nlp_model, embedding_model, device, **settings):
        nlp = spacy.load(nlp_model)
        encoder = SentenceTransformer(embedding_model, device=str(device), local_files_only=True)
        return cls(judge, nlp, encoder, **settings)

    @torch.inference_mode()
    def __call__(self, caption, objects, focus):
        sentences = [s.strip() for s in re.split(r"\n|\\n|</s>|[.]", caption) if s.strip()]
        nouns, sentence_ids = [], []
        for index, doc in enumerate(self.nlp.pipe(sentences)):
            for token in doc:
                if token.tag_.startswith("N"):
                    nouns.append(token.lemma_.lower())
                    sentence_ids.append(index)
        if not nouns or not objects:
            return 0.0

        names = list(dict.fromkeys(obj["category"] for obj in objects))
        embeddings = self.encoder.encode(
            names + nouns, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )
        matrix = embeddings[:len(names)] @ embeddings[len(names):].T
        similarities = dict(zip(names, matrix.cpu()))
        weights = self.weights[focus]
        regions = [region for region in ("foreground", "background") if weights[region]]
        if not regions:
            return self.score_group(objects, similarities, sentences, sentence_ids, weights, regional=False)
        reward = 0.0
        for region in regions:
            group = [obj for obj in objects if obj["tag"] == region]
            if group:
                score = self.score_group(group, similarities, sentences, sentence_ids, weights, regional=True)
                reward += weights[region] * score
        return reward

    def score_group(self, objects, similarities, sentences, sentence_ids, weights, regional):
        categories = list(dict.fromkeys(obj["category"] for obj in objects))
        sim = torch.stack([similarities[name] for name in categories])
        best = sim.amax(dim=1)
        matches = (sim == best[:, None]) & (sim == sim.amax(dim=0)) & (sim > self.similarity_threshold)
        covered = matches.any(dim=1).float()
        if self.soft_coverage:
            covered *= best
        coverage = covered.mean().item() * self.object_scale
        evidence = {
            category: {sentence_ids[j] for j in row.nonzero().flatten().tolist()}
            for category, row in zip(categories, matches)
        }
        attributes, relations = judge_comparisons(objects, evidence, sentences)
        scores = self.judge.score(attributes + relations)
        attribute = sum(scores[:len(attributes)]) / (5 * len(attributes))
        relation = sum(scores[len(attributes):]) / (5 * len(relations)) if relations else 0.0

        obj_weight, attr_weight = weights["object"], weights["attribute"]
        if regional and not relations:
            total = obj_weight + attr_weight
            if total > 0:
                obj_weight, attr_weight = obj_weight / total, attr_weight / total
            else:
                obj_weight = attr_weight = 0.5
        return obj_weight * coverage + attr_weight * attribute + weights["relation"] * relation
