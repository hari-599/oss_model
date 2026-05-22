import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatAssistant:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.history = []
        self.system_prompt = (
            "You are a helpful, safe, and concise AI assistant. "
            "Maintain conversational context and avoid harmful responses."
        )


    def build_message(self, user_message: str) -> str:
        messages=[
            {"role": "system", "content": self.system_prompt},
        ]
        for turn in self.history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
            messages.append({"role": "user", "content": user_message})
        return messages

    def generate_response(self, user_message: str) -> str:
        prompt = self.build_message(user_message)
        text=self.tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.75,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        self.history.append({"user": user_message, "assistant": response})
        return response
