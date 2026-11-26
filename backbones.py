import random
from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPImageProcessor, Qwen2_5_VLForConditionalGeneration


class QwenBackbone:
    def __init__(self, path, processor_path, model_kwargs, processor_kwargs):
        self.path = path
        self.model_kwargs = model_kwargs
        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True, **processor_kwargs)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def load_model(self, gradient_checkpointing):
        settings = self.model_kwargs.copy()
        if gradient_checkpointing:
            settings["use_cache"] = False
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(self.path, local_files_only=True, **settings)

    def encode(self, images, prompts, training, max_prompt_length):
        texts = []
        for prompt in prompts:
            content = [{"type": "text", "text": prompt}, {"type": "image"}]
            texts.append(self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                           tokenize=False, add_generation_prompt=True))
        return self.processor(text=texts, images=images, padding=True, padding_side="left", return_tensors="pt")

    def generate(self, model, batch, config):
        tokens = model.generate(**batch, generation_config=config, tokenizer=self.tokenizer)
        return tokens[:, batch["input_ids"].size(1):]


def image_tiles(image, size, max_tiles, thumbnail):
    grids = sorted({(x, y) for x in range(1, max_tiles + 1) for y in range(1, max_tiles // x + 1)},
                   key=lambda grid: grid[0] * grid[1])
    aspect = image.width / image.height
    best_diff = float("inf")
    for cols, rows in grids:
        diff = abs(aspect - cols / rows)
        if diff < best_diff or (diff == best_diff and image.width * image.height > size * size * cols * rows / 2):
            grid = cols, rows
            best_diff = diff
    cols, rows = grid
    resized = image.resize((cols * size, rows * size))
    tiles = [resized.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
             for y in range(rows) for x in range(cols)]
    if thumbnail and len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    return tiles


def jpeg_image(image, **options):
    with BytesIO() as stream:
        image.save(stream, format="JPEG", **options)
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert("RGB")


class InternVLBackbone:
    def __init__(self, path, processor_path, model_kwargs, processor_kwargs):
        self.path = path
        self.model_kwargs = model_kwargs
        self.processor = AutoTokenizer.from_pretrained(processor_path, local_files_only=True, use_fast=False,
                                                      add_eos_token=False, trust_remote_code=model_kwargs.get("trust_remote_code", False))
        self.tokenizer = self.processor
        self.tokenizer.padding_side = "left"
        self.image_processor = CLIPImageProcessor.from_pretrained(processor_path, local_files_only=True)
        self.new_tokens = self.tokenizer.add_tokens(["<img>", "</img>", "<IMG_CONTEXT>", "<quad>", "</quad>",
                                                    "<ref>", "</ref>", "<box>", "</box>"], special_tokens=True)
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.max_tiles = processor_kwargs["max_tiles"]
        self.thumbnail = processor_kwargs["thumbnail"]
        self.jpeg_quality = processor_kwargs.get("jpeg_quality")
        if self.max_tiles < 1:
            raise ValueError("InternVL max_tiles must be positive")

    def load_model(self, gradient_checkpointing):
        model = AutoModel.from_pretrained(self.path, local_files_only=True, **self.model_kwargs)
        model.register_for_auto_class("AutoModel")
        model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if self.new_tokens:
            model.language_model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                weights = model.language_model.get_output_embeddings().weight
                weights[-self.new_tokens:] = weights[:-self.new_tokens].mean(dim=0)
            model.config.llm_config.vocab_size = len(self.tokenizer)
        model.language_model.config.use_cache = False
        model.vision_model.gradient_checkpointing = gradient_checkpointing
        model.vision_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.template = model.conv_template
        self.num_image_tokens = model.num_image_token
        self.image_size = model.config.force_image_size or model.config.vision_config.image_size
        return model

    def encode(self, images, prompts, training, max_prompt_length):
        pixels, texts = [], []
        for image, prompt in zip(images, prompts):
            tiles = image_tiles(jpeg_image(image), self.image_size, self.max_tiles, self.thumbnail)
            if training and self.jpeg_quality:
                low, high = self.jpeg_quality
                tiles = [jpeg_image(tile, quality=random.choices(range(low, high + 1))[0]) for tile in tiles]
            pixels.extend(to_tensor(tile) for tile in tiles)
            template = self.template.copy()
            image_text = "<img>" + "<IMG_CONTEXT>" * (len(tiles) * self.num_image_tokens) + "</img>"
            template.append_message(template.roles[0], f"{image_text}\n{prompt}")
            texts.append(template.get_prompt())
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_prompt_length,
                               return_token_type_ids=False, return_tensors="pt")
        batch["pixel_values"] = normalize(torch.stack(pixels), self.image_processor.image_mean, self.image_processor.image_std)
        batch["image_flags"] = torch.ones(len(pixels), dtype=torch.long)
        return batch

    def generate(self, model, batch, config):
        return model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              pixel_values=batch["pixel_values"], generation_config=config, tokenizer=self.tokenizer)


def load_backbone(family, path, model_kwargs, processor_kwargs, processor_path=None):
    families = {"qwen2_5_vl": QwenBackbone, "internvl": InternVLBackbone}
    return families[family](path, processor_path or path, model_kwargs, processor_kwargs)
