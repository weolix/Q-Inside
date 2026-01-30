# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainerComparison, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    score_reward_threshold: Optional[float] = field(
        default=0.35,
        metadata={"help": "Threshold for score reward"},
    )
    dataset_dist: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file path for the distortion detection dataset"},
    )
    dataset_score: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file path for the quality scoring dataset"},
    )
    dataset_comparison: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file path for the comparison dataset"},
    )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

COMPARE_QUESTION_PROMPT =     (
    "Decide which enhanced image is superior "
    "or if they are comparable. Evaluate based on: "
    "1) fidelity and consistency with the reference image; "
    "2) overall perceptual quality. "
    "Return **exactly one** of: Image A, Image B, or Similar."
)



class LazyComparisonDataset(Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args

        # Only supports datasets for a single image comparison task
        yaml_path = getattr(script_args, "dataset_comparison", None)
        if not yaml_path:
            raise ValueError("Please provide the dataset file: --dataset_comparison <path_to_yaml>")
        self.samples = self._load_samples_from_yaml(yaml_path)
        if not self.samples:
            raise ValueError("No samples loaded; please check your dataset file content and path.")
        self.total_len = len(self.samples)

    def _load_samples_from_yaml(self, data_path: str):
        samples = []
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r") as f:
            cfg = yaml.safe_load(f)
            for ds in cfg.get("datasets", []):
                path = ds.get("json_path")
                strategy = ds.get("sampling_strategy", "all")
                # load JSON or JSONL
                if path.endswith(".jsonl"):
                    data_list = [json.loads(line) for line in open(path, "r")]
                elif path.endswith(".json"):
                    data_list = json.load(open(path, "r"))
                else:
                    raise ValueError(f"Unsupported file type: {path}")
                # sampling
                if ":" in strategy:
                    strat, num_s = strategy.split(":")
                    if "%" in num_s:
                        count = math.ceil(int(num_s.rstrip("%")) * len(data_list) / 100)
                    else:
                        count = int(num_s)
                    strategy = strat
                if strategy == "first":
                    data_list = data_list[:count]
                elif strategy == "end":
                    data_list = data_list[-count:]
                elif strategy == "random":
                    random.shuffle(data_list)
                    data_list = data_list[:count]
                print(f"Loaded {len(data_list)} samples from {path}")
                samples.extend(data_list)
        return samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        example = self.samples[index]
        sample = {}
        # 1. ground truth solution
        sample['solution'] = example.get('result')

        root = self.script_args.image_root
        # 2. Load and attach PIL.Image objects for generate()
        # Reference image
        ref_rel = example.get('ref_image')
        ref_fp = os.path.join(root, ref_rel)
        if not os.path.exists(ref_fp):
            raise FileNotFoundError(f"Reference image not found: {ref_fp}")
        sample['ref_image'] = Image.open(ref_fp).convert('RGB')
        sample['ref_image_path'] = ref_fp
        # Image A
        A_rel = example.get('ImageA')
        A_fp  = os.path.join(root, A_rel)
        if not os.path.exists(A_fp):
            raise FileNotFoundError(f"Image A not found: {A_fp}")
        sample['imageA'] = Image.open(A_fp).convert('RGB')
        sample['imageA_path'] = A_fp
        # Image B
        B_rel = example.get('ImageB')
        B_fp  = os.path.join(root, B_rel)
        if not os.path.exists(B_fp):
            raise FileNotFoundError(f"Image B not found: {B_fp}")
        sample['imageB'] = Image.open(B_fp).convert('RGB')
        sample['imageB_path'] = B_fp

        # 3. Prompt fields
        sample['system_prompt']   = example.get('system_prompt', SYSTEM_PROMPT)
        sample['custom_question'] = example.get('custom_question', COMPARE_QUESTION_PROMPT)

        return sample





def score_reward(completions, solution, **kwargs):
    """
    For comparison tasks only:
      - Extract text from the <answer> tag.
      - If it exactly matches the solution (e.g., "Image A" or "Image B"), assign a reward of 1.0; otherwise, 0.0.
      - Preserve DEBUG logs by writing each match result to a file.
    """
    contents = [c[0]["content"] for c in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'

    for idx, (content, true_sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        answer_text = ""
        try:
            m = re.search(answer_tag_pattern, content, re.DOTALL)
            if m:
                answer_text = m.group(1).strip()
                pat = re.compile(rf"^{re.escape(true_sol)}$")
                if pat.fullmatch(answer_text):
                    reward = 1.0
        except Exception as e:
            print(f"Error in computing comparison reward at idx {idx}:", e)

        rewards.append(reward)

        # DEBUG logging
        if os.getenv("DEBUG_MODE") == "true":
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            log_path = os.getenv("LOG_PATH", "comparison_reward.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"----- {now} Rank:{rank} Index:{idx} -----\n")
                f.write(f"Expected: {true_sol!r}\n")
                f.write(f"Answer:   {answer_text!r}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Reward:   {reward}\n\n")

    return rewards


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    and the final answer is enclosed within <answer> and </answer> tags.
    In addition, the content inside <answer> (after stripping leading/trailing whitespace)
    must be a JSON-like string where the first non-whitespace character is '{' and the last is '}',
    and no extra '{' or '}' appear inside.
    """
    pattern = (
        r"^<think>\s*\n"         # <think> tag, possibly with whitespace, then a newline
        r"[\s\S]*?\n"            # Content inside <think> (non-greedy), including newlines
        r"\s*</think>\s*\n"      # </think> tag, possibly with whitespace, then a newline
        r"<answer>\s*\n"         # <answer> tag, possibly with whitespace, then a newline
        r"[\s\S]*?\n"            # Content inside <answer> (any characters, including newlines, non-greedy)
        r"\s*</answer>\s*$"      # </answer> tag, possibly with whitespace, until end of string
    )
    
    completion_contents = [completion[0]["content"] for completion in completions]

    # Use re.fullmatch to ensure the entire string matches the regex pattern
    matches = [re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]




reward_funcs_registry = {
    "accuracy": score_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazyComparisonDataset(script_args)

    trainer_cls = Qwen2VLGRPOTrainerComparison
    
    # Prepare PEFT/LoRA config
    peft_cfg = get_peft_config(model_args)
    if peft_cfg is None:
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        print("Using default LoRA config for PEFT (r=8, alpha=32, dropout=0.1)")

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_cfg,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
