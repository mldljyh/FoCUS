import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from accelerate.utils import broadcast
from torch.nn.functional import pad
from transformers import GenerationConfig, Trainer
from transformers.integrations import is_deepspeed_zero3_enabled
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax


def collate_scenes(rows):
    return {"scenes": rows}


class CaptionGRPOTrainer(Trainer):
    def __init__(self, backbone, reward, prompts, prompt_cycle, rollout, **kwargs):
        self.backbone = backbone
        self.reward = reward
        self.prompts = prompts
        self.cycle = prompt_cycle
        self.rollout = rollout
        if rollout["num_generations"] < 2:
            raise ValueError("GRPO requires at least two generations per training prompt")
        if rollout["advantage_epsilon"] <= 0:
            raise ValueError("advantage_epsilon must be positive")
        if not self.cycle or set(self.cycle) != set(prompts) or set(prompts) != set(reward.weights):
            raise ValueError("Prompts, prompt_cycle and reward weights must cover the same focus types")
        self.batch_index = 0
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        self.eval_completions = []
        checkpointing = kwargs["args"].gradient_checkpointing
        model = backbone.load_model(checkpointing)
        reference = None
        if rollout["beta"]:
            reference = backbone.load_model(checkpointing) if is_deepspeed_zero3_enabled() else create_reference_model(model)
            reference.requires_grad_(False).eval()
        super().__init__(model=model, processing_class=backbone.processor, data_collator=collate_scenes, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.ref_model = None
        if reference is not None:
            plugin = self.accelerator.state.deepspeed_plugin
            if plugin is None:
                self.ref_model = self.accelerator.prepare_model(reference, evaluation_mode=True)
            elif plugin.zero_stage == 3:
                self.ref_model = prepare_deepspeed(reference, self.accelerator)
            else:
                config = plugin.hf_ds_config
                dtype = config.dtype() if config.is_true("bf16.enabled") or config.is_true("fp16.enabled") else None
                self.ref_model = reference.to(device=self.accelerator.device, dtype=dtype)
                broadcast(list(self.ref_model.parameters()))
        token_ids = {"pad_token_id": backbone.tokenizer.pad_token_id}
        self.train_generation = GenerationConfig(**(token_ids | rollout["generation"]))
        self.eval_generation = GenerationConfig(**(token_ids | rollout["eval_generation"]))
        if self.train_generation.num_return_sequences != 1 or self.eval_generation.num_return_sequences != 1:
            raise ValueError("Use num_generations for group size and leave num_return_sequences at one")

    def training_focuses(self, batch_size):
        start = self.batch_index + self.accelerator.process_index
        self.batch_index += 1
        groups = []
        for i in range(batch_size):
            focus = self.cycle[(start + i) % len(self.cycle)]
            groups.append([focus] * self.rollout["num_generations"])
        return groups

    def completion_targets(self, tokens, limit):
        pad_id = self.backbone.pad_id
        eos_id = self.backbone.eos_id
        tokens = tokens[:, :limit]
        tokens = pad(tokens, (0, limit - tokens.size(1)), value=pad_id)
        positions = torch.arange(limit, device=tokens.device)
        stops = tokens.eq(eos_id) | tokens.eq(pad_id)
        end = torch.where(stops, positions, limit).amin(dim=1, keepdim=True)
        targets = tokens.masked_fill(positions > end, pad_id).masked_fill(positions == end, eos_id)
        return targets, (positions <= end).float()

    @torch.no_grad()
    def sample(self, model, scenes, groups, training):
        images, prompts, targets = [], [], []
        for scene, group in zip(scenes, groups):
            for focus in group:
                images.append(scene["image"])
                prompts.append(self.prompts[focus])
                targets.append((scene["objects"], focus))
        limit = self.rollout["max_prompt_length"]
        batch = self._prepare_inputs(dict(self.backbone.encode(images, prompts, training, limit)))
        if batch["input_ids"].size(1) > limit:
            warnings.warn(f"Prompt length {batch['input_ids'].size(1)} exceeds max_prompt_length {limit}", stacklevel=2)
        batch["pixel_values"] = batch["pixel_values"].to(dtype=self.model.dtype)
        generation = self.train_generation if training else self.eval_generation
        with unwrap_model_for_generation(model, self.accelerator) as policy:
            tokens = self.backbone.generate(policy, batch, generation)
        tokens, mask = self.completion_targets(tokens, generation.max_new_tokens)
        captions = self.backbone.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        rewards = [self.reward(caption, objects, focus) for caption, (objects, focus) in zip(captions, targets)]
        scores = torch.tensor(rewards, device=tokens.device, dtype=torch.float32)
        return batch, tokens, mask, scores, captions

    def record_rewards(self, scores, groups, metrics):
        rewards = self.accelerator.gather(scores)
        focus_ids = scores.new_tensor([self.cycle.index(focus) for group in groups for focus in group], dtype=torch.long)
        focuses = self.accelerator.gather(focus_ids)
        size = self.rollout["num_generations"]
        full = len(rewards) // size * size
        grouped = rewards[:full].view(-1, size)
        means = grouped.mean(dim=1)
        stds = grouped.std(dim=1) if full else rewards[:0]
        if full < len(rewards):
            tail = rewards[full:]
            means = torch.cat([means, tail.mean().view(1)])
            tail_std = tail.std() if len(tail) > 1 else tail.new_ones(())
            stds = torch.cat([stds, tail_std.view(1)])
        means = means.repeat_interleave(size)[:len(rewards)]
        stds = torch.where(stds.isnan() | stds.eq(0), 1.0, stds).repeat_interleave(size)[:len(rewards)]
        advantages = (rewards - means) / (stds + self.rollout["advantage_epsilon"])
        if self.rollout["flat_batch_advantage_std"] and torch.all(advantages == 0):
            advantages = torch.randn_like(advantages) * self.rollout["flat_batch_advantage_std"]

        metrics["default/rewards/llm_judge_reward_function"].append(scores.mean().item())
        metrics["reward"].append(rewards.mean().item())
        metrics["reward_std"].append(stds.mean().item())
        for index in focuses.unique().tolist():
            values = rewards[focuses == index].tolist()
            metrics[f"reward/{self.cycle[index]}"].append(sum(values) / len(values))
        return advantages.narrow(0, self.accelerator.process_index * len(scores), len(scores))

    @staticmethod
    def token_logps(model, batch, completion):
        logits = model(**batch, use_cache=False).logits[:, -completion.size(1) - 1:-1]
        return selective_log_softmax(logits, completion)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scenes = inputs["scenes"]
        groups = self.training_focuses(len(scenes))
        batch, tokens, mask, rewards, _ = self.sample(model, scenes, groups, True)
        batch["input_ids"] = torch.cat([batch["input_ids"], tokens], dim=1)
        prompt_mask = batch["attention_mask"]
        reference_logps = None
        if self.ref_model is not None:
            batch["attention_mask"] = torch.cat([prompt_mask, mask], dim=1)
            with torch.inference_mode():
                reference_logps = self.token_logps(self.ref_model, batch, tokens)
        advantages = self.record_rewards(rewards, groups, self.train_metrics)
        batch["attention_mask"] = torch.cat([prompt_mask, mask.int()], dim=1)
        logps = self.token_logps(model, batch, tokens)
        lengths = mask.sum(dim=1)
        kl = None
        if reference_logps is not None:
            kl = (reference_logps - logps).exp() - (reference_logps - logps) - 1
        objective = (logps - logps.detach()).exp() * advantages[:, None]
        if kl is not None:
            objective = objective - self.rollout["beta"] * kl
            mean_kl = ((kl.detach() * mask).sum(1) / lengths).mean()
            self.train_metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        loss = ((-objective * mask).sum(1) / lengths).mean()
        self.train_metrics["completion_length"].append(self.accelerator.gather_for_metrics(lengths).mean().item())
        return (loss, {"rewards": rewards.view(len(scenes), -1)}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        scenes = inputs["scenes"]
        groups = [[self.cycle[i % len(self.cycle)]] for i in range(len(scenes))]
        with self.compute_loss_context_manager():
            _, _, _, scores, captions = self.sample(model, scenes, groups, False)
        self.record_rewards(scores, groups, self.eval_metrics)
        if self.rollout.get("log_completions"):
            for group, caption, score in zip(groups, captions, scores.tolist()):
                self.eval_completions.append({"question": self.prompts[group[0]], "prediction": caption,
                                             "reward_names": ["llm_judge_reward_function"], "rewards": [score]})
        return scores.new_zeros(()), None, None

    @staticmethod
    def metric_means(metrics, prefix=""):
        means = {name: sum(values) / len(values) for name, values in metrics.items()}
        for name, value in list(means.items()):
            if name in ("reward", "reward_std"):
                means[f"default/{name}"] = value
            elif name.startswith("reward/"):
                means[f"default/reward_by_focus/{name.split('/', 1)[1]}"] = value
        metrics.clear()
        return {prefix + name: value for name, value in means.items()}

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        self.eval_metrics.clear()
        self.eval_completions.clear()
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        output.metrics.update(self.metric_means(self.eval_metrics, metric_key_prefix + "_"))
        if self.eval_completions:
            directory = Path(self.args.output_dir) / f"local_log_step_{self.state.global_step}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"evaluation_results_{self.accelerator.process_index}.json"
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(previous + self.eval_completions, indent=2), encoding="utf-8")
            temporary.replace(path)
            self.eval_completions.clear()
        return output

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs = {**logs, **self.metric_means(self.train_metrics)}
        super().log(logs, start_time)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if hasattr(self.backbone, "image_processor"):
            self.backbone.image_processor.save_pretrained(output_dir or self.args.output_dir)
