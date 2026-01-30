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
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
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

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
SCORE_QUESTION_PROMPT = "Please answer with: The quality of this image is [quality description]."
DIST_QUESTION_PROMPT = 'Analyze the given image and determine if it contains any of the following distortions: "noise", "compression", "blur", or "darken". If a distortion is present, classify its severity as "slight", "moderate", "obvious", "serious", or "catastrophic". Return the result in JSON format with the following keys: "distortion_class": The detected distortion (or "null" if none). and "severity": The severity level (or "null" if none).'


class LazySupervisedDataset(Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args

        # Load datasets for the two types of tasks separately
        self.dist_samples = []
        self.score_samples = []
        if script_args.dataset_dist:
            self.dist_samples = self._load_samples_from_yaml(script_args.dataset_dist)
        if script_args.dataset_score:
            self.score_samples = self._load_samples_from_yaml(script_args.dataset_score)

        if not self.dist_samples and not self.score_samples:
            raise ValueError("At least one dataset file must be provided: --dataset_dist or --dataset_score")
        
        # Return the total number of samples (sum of both task datasets)
        self.total_len = len(self.dist_samples) + len(self.score_samples)

    def _load_samples_from_yaml(self, data_path: str):
        """
        Load sample data from a given YAML file.
        Example format of the YAML file:
          datasets:
            - json_path: xxxx1.json
              sampling_strategy: first:1000
            - json_path: xxxx2.json
              sampling_strategy: end:3000
            - json_path: xxxx3.json
              sampling_strategy: random:999
        """
        samples = []
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets", [])
            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                samples.extend(cur_data_dict)
        return samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        """
        Return a sample from the merged dataset based on the index and determine which task it belongs to:
        - If the index is less than the number of score_samples, the sample comes from the scoring task (score), using gt_score_norm as the solution;
        - Otherwise, the sample comes from the degradation detection task (dist), with the solution containing distortion_class and severity.
        """        
        # If both score_samples and dist_samples exist, use simple concatenation:
        if self.score_samples and self.dist_samples:
            if index < len(self.score_samples):
                chosen_task = "score"
                example = self.score_samples[index]
                solution = example.get("gt_score_norm", None)
                prompt_text = SCORE_QUESTION_PROMPT  # Prompt for the scoring task
            else:
                chosen_task = "dist"
                # For dist_samples index, subtract the length of score_samples
                index2 = index - len(self.score_samples)
                if index2 >= len(self.dist_samples):
                    raise IndexError("Index out of range for dist_samples")
                example = self.dist_samples[index2]
                solution = {
                    "distortion_class": example.get("distortion_class", None),
                    "severity": example.get("severity", None)
                }
                prompt_text = DIST_QUESTION_PROMPT  # Prompt for the degradation detection task

        # If only score_samples exists:
        elif self.score_samples:
            chosen_task = "score"
            example = self.score_samples[index]
            solution = example.get("gt_score_norm", None)
            prompt_text = SCORE_QUESTION_PROMPT

        # If only dist_samples exists:
        elif self.dist_samples:
            chosen_task = "dist"
            example = self.dist_samples[index]
            solution = {
                "distortion_class": example.get("distortion_class", None),
                "severity": example.get("severity", None)
            }
            prompt_text = DIST_QUESTION_PROMPT

        else:
            raise ValueError("No available dataset (score_samples or dist_samples)")

        sample = {"task": chosen_task, "solution": solution}

        # Process the image
        image = None
        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            # If the image path does not exist, try randomly selecting another sample for the corresponding task
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, trying another sample")
                if chosen_task == "score":
                    new_index = random.randint(0, len(self.score_samples) - 1)
                    example = self.score_samples[new_index]
                else:
                    new_index = random.randint(0, len(self.dist_samples) - 1)
                    example = self.dist_samples[new_index]
                image_path = os.path.join(image_root, example["image"])
            image = Image.open(image_path).convert("RGB")
        sample["image"] = image
        sample["image_path"] = image_path

        # Construct the prompt to maintain consistency with system and user roles
        sample["prompt"] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
        ]

        return sample



def score_reward(completions, solution, task, image_path, pred_score=None, **kwargs):
    """
    Compute the reward based on the format and content of the generated answers.

    For the 'score' task:
      - Use the regression score derived from logits (pred_score) if available.
      - If the model’s score differs from the ground truth (gt_score_norm) by less than the threshold (default 0.35), reward = 1.0.

    For the 'dist' task:
      - Extract the JSON string from the <answer> tag and match the "distortion_class" and "severity" fields.
      - If the model’s distortion_class matches the ground truth, add 0.25 to the reward.
      - If the severity also matches, add an additional 0.75.
    """
    # Extract the content from each generated answer
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # Define regular expression patterns
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    score_pattern = r'\"rating\"\s*:\s*([\d\.]+)'
    dist_pattern = r'\"distortion_class\"\s*:\s*\"([^\"]+)\".*?\"severity\"\s*:\s*\"([^\"]+)\"'
    
    # Compute the sampling ratio to align 'task' length with number of completions
    num_gen = len(task) // len(contents)
    subsampled_tasks = task[::num_gen]
    subsampled_solutions = solution[::num_gen]
    subsampled_image_paths = image_path[::num_gen]
    subsampled_pred_scores = pred_score[::num_gen] if pred_score is not None else None

    score_reward_threshold = script_args.score_reward_threshold
    
    model_score = None
    model_distortion_class = None
    model_severity = None
    for i, (t, content, true_sol) in enumerate(zip(subsampled_tasks, contents, subsampled_solutions)):
        reward = 0.0
        try:
            # Extract the answer content from the <answer> tag
            match_answer = re.search(answer_tag_pattern, content, re.DOTALL)
            if match_answer:
                answer_content = match_answer.group(1).strip()
                if t == 'score':
                    model_score = None
                    if subsampled_pred_scores is not None:
                        model_score = float(subsampled_pred_scores[i])
                    else:
                        # Fallback to parse numeric score from JSON if provided
                        match_score = re.search(score_pattern, answer_content)
                        if match_score:
                            model_score = float(match_score.group(1))
                    if model_score is not None and abs(model_score - true_sol) < score_reward_threshold:
                        reward = 1.0
                elif t == 'dist':
                    # For degradation detection tasks, match "distortion_class" and "severity"
                    match_dist = re.search(dist_pattern, answer_content, re.DOTALL)
                    if match_dist:
                        model_distortion_class = match_dist.group(1).strip()
                        model_severity = match_dist.group(2).strip()
                        # true_sol is expected to be a dict containing distortion_class and severity
                        gt_distortion_class = true_sol.get("distortion_class", None)
                        gt_severity = true_sol.get("severity", None)

                        if model_distortion_class == gt_distortion_class:
                            reward += 0.25
                            if model_severity == gt_severity:
                                reward += 0.75
        except Exception as e:
            print("Error in computing reward", e)
        rewards.append(reward)


        if os.getenv("DEBUG_MODE") == "true":
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                current_rank = torch.distributed.get_rank()
            else:
                current_rank = 0
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Rank: {current_rank} Task: {t} Reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Image Path: {subsampled_image_paths[i]}\n")
                if t == 'score':
                    try:
                        if model_score is not None:
                            f.write(f"Model Score: {model_score}\n")
                    except Exception as e:
                        f.write("Write Model Score Error!\n")
                elif t == 'dist':
                    try:
                        if model_distortion_class is not None:
                            f.write(f"Model Distortion Class: {model_distortion_class}\n")
                        if model_severity is not None:
                            f.write(f"Model Severity: {model_severity}\n")
                    except Exception as e:
                        f.write("Write Model Dist Error!\n")
                f.write(f"Ground Truth: {true_sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    and the final answer is enclosed within <answer> and </answer> tags.
    In addition, the content inside <answer> must start with "The quality of this image is".
    """
    pattern = (
        r"^<think>\s*\n"         # <think> tag, optional whitespace, then newline
        r".*?\n"                 # content of think (non-greedy) until a newline
        r"\s*</think>\s*\n"      # closing </think> tag with optional whitespace then newline
        r"<answer>\s*\n"         # <answer> tag with optional whitespace then newline
        r"The quality of this image is[\s\S]*?"
        r"\s*</answer>\s*$"      # closing </answer> tag with optional whitespace until end of string
    )
    
    completion_contents = [completion[0]["content"] for completion in completions]
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
    dataset = LazySupervisedDataset(script_args)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
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
