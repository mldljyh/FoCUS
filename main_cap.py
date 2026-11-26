import argparse
from pathlib import Path

import yaml


def train(config, resume=None, judge_base_url=None):
    from transformers import TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    from backbones import load_backbone
    from data import load_datasets
    from reward_cap import CaptionReward
    from trainer import CaptionGRPOTrainer
    from vllm_judge import VLLMJudge

    args = TrainingArguments(**(config["training"] | {"remove_unused_columns": False}))
    set_seed(args.seed)
    train_data, eval_data = load_datasets(**config["data"])
    if args.eval_strategy != "no" and eval_data is None:
        raise ValueError("Evaluation requires a nonempty validation split")
    judge_settings = config["judge"].copy()
    if judge_base_url:
        judge_settings["base_url"] = judge_base_url
    judge = VLLMJudge(**judge_settings)
    try:
        judge.check_ready()
        settings = config["reward"].copy()
        if settings["device"] == "training":
            settings["device"] = args.device
        reward = CaptionReward.from_config(judge, **settings)
        trainer = CaptionGRPOTrainer(
            backbone=load_backbone(**config["model"]),
            reward=reward,
            prompts=config["prompts"],
            prompt_cycle=config["prompt_cycle"],
            rollout=config["rollout"],
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
        )
        checkpoint = resume if resume is not None else args.resume_from_checkpoint
        if checkpoint == "auto":
            checkpoint = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
    finally:
        judge.client.close()


def main():
    parser = argparse.ArgumentParser(description="Train a FoCUS captioning policy with a running vLLM judge")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--judge-base-url", help="Override the configured OpenAI-compatible /v1 URL")
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", help="Checkpoint directory or auto")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    from torch import distributed

    owns_group = not distributed.is_initialized()
    try:
        train(config, args.resume_from_checkpoint, args.judge_base_url)
    finally:
        if owns_group and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
