import os
import random
import sys
import torch
import math
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
# Get the parent directory of 'comfy' and add it to the Python path
comfy_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(comfy_parent_dir)

# Suppress console output
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Import the required modules
import comfy.model_management as model_management
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from comfy.model_patcher import ModelPatcher
from .util import join_prompts, remove_empty_str

# Restore the original stdout
sys.stdout = original_stdout

fooocus_expansion_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                      'fooocus_expansion'))
fooocus_magic_split = [
    ', extremely',
    ', intricate,',
]
dangerous_patterns = '[]【】()（）|:：'

neg_inf = -8192.0


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x
class FooocusExpansion:
    SEED_LIMIT_NUMPY = 1125899906842624

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_path)

        positive_words = open(os.path.join(fooocus_expansion_path, 'positive.txt'),
                              encoding='utf-8').read().splitlines()
        positive_words = ['Ġ' + x.lower() for x in positive_words if x != '']

        self.logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])


        # debug_list = '\n'.join(sorted(debug_list))
        # print(debug_list)

        # t11 = self.tokenizer(',', return_tensors="np")
        # t198 = self.tokenizer('\n', return_tensors="np")
        # eos = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_path)
        self.model.eval()

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()

        # MPS hack
        if model_management.is_device_mps(load_device):
            load_device = torch.device('cpu')
            offload_device = torch.device('cpu')

        use_fp16 = model_management.should_use_fp16(device=load_device)

        if use_fp16:
            self.model.half()

        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)
       

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0

        return scores + bias

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt, seed):
        if prompt == '':
            return ''

        if self.patcher.current_device != self.patcher.load_device:
           
            model_management.load_model_gpu(self.patcher)

        seed = int(seed) % self.SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = safe_str(prompt) + ','

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.patcher.load_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.patcher.load_device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=LogitsProcessorList([self.logits_processor]))

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result
    
class PromptExpansion:
    # Define the expected input types for the node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                "log_prompt": (["No", "Yes"], {"default": "No"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("final_prompt", "seed",)
    FUNCTION = "expand_prompt"  # Function name

    CATEGORY = "utils"  # Category for organization

    @staticmethod
    @torch.no_grad()
    def expand_prompt(text, seed, log_prompt):
        expansion = FooocusExpansion()

        prompt = remove_empty_str([safe_str(text)], default='')[0]

        max_seed = FooocusExpansion.SEED_LIMIT_NUMPY
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed = seed % (2**32 - 1)

        final_prompt = expansion(prompt, seed)

        if log_prompt == "Yes":
           
            print(f"Final prompt: {final_prompt}")

        return final_prompt, seed


# Define a mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "PromptExpansion": PromptExpansion
}

# A dictionary that contains human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptExpansion": "Prompt Expansion"
}