import sys
sys.path.append("indobenchmark-toolkit/src")

import os
os.environ["HF_HOME"] = "./hf_cache"

from indobenchmark import IndoNLGTokenizer
from transformers import GPT2LMHeadModel, BatchEncoding

class IndoNLGTokenizerForChat(IndoNLGTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.system_turn_token = "[system]"
        self.user_turn_token = "[user]"
        self.assistant_turn_token = "[assistant]"
        self.newline_turn_token = "[newline]"

        next_new_token_id = max(self.special_tokens_to_ids.values()) + 1
        for new_token in (
                self.system_turn_token,
                self.user_turn_token,
                self.assistant_turn_token,
                self.newline_turn_token,
        ):
            self.special_tokens_to_ids[new_token] = next_new_token_id
            self.special_ids_to_tokens[next_new_token_id] = new_token
            self.special_token_ids.append(next_new_token_id)
            next_new_token_id += 1

        self.system_turn_token_id = self.special_tokens_to_ids[self.system_turn_token]
        self.user_turn_token_id = self.special_tokens_to_ids[self.user_turn_token]
        self.assistant_turn_token_id = self.special_tokens_to_ids[self.assistant_turn_token]
        self.newline_turn_token_id = self.special_tokens_to_ids[self.newline_turn_token]

    def prepare_inputs_for_training(self, instructions: list[str], user_chats: list[str], assistant_chats: list[str]):
        # We assume it's just a simple single chat.
        assert len(instructions) == len(user_chats) == len(assistant_chats)
        instructions_inputs = self._prepare_instruction_inputs(instructions)
        user_inputs = self._prepare_user_inputs(user_chats)
        assistant_inputs = self._prepare_assistant_inputs(assistant_chats)
        inputs = BatchEncoding({
            k: [x + y + z for x, y, z in zip(
                instructions_inputs[k],
                user_inputs[k],
                assistant_inputs[k]
            )] for k in instructions_inputs.keys()
        })
        return inputs

    def _prepare_instruction_inputs(self, instructions: list[str]):
        inputs = self([f"<s>{inst}</s>" for inst in instructions])
        self._insert_token_id_at_beginning_of_inputs(inputs, self.system_turn_token_id)
        self._make_mask_labels_from_input_ids_in_inputs(inputs)
        return inputs

    def _prepare_user_inputs(self, user_chats: list[str]):
        inputs = self([f"<s>{uc}</s>" for uc in user_chats])
        self._insert_token_id_at_beginning_of_inputs(inputs, self.user_turn_token_id)
        self._make_mask_labels_from_input_ids_in_inputs(inputs)
        return inputs
    
    def _prepare_assistant_inputs(self, assistant_chats: list[str]):
        prefix_inputs = self(["<s>" for _ in assistant_chats])
        self._insert_token_id_at_beginning_of_inputs(prefix_inputs, self.assistant_turn_token_id)
        self._make_mask_labels_from_input_ids_in_inputs(prefix_inputs)

        response_inputs = self([f"{ac} </s>" for ac in assistant_chats])
        response_input_ids: list[list[int]] = response_inputs["input_ids"]
        response_inputs["labels"] = [[y for y in x] for x in response_input_ids]

        return BatchEncoding({
            k: [x + y for x, y in zip(prefix_inputs[k], response_inputs[k])]
            for k in prefix_inputs.keys()
        })
    
    def _make_mask_labels_from_input_ids_in_inputs(self, inputs):
        input_ids: list[list[int]] = inputs["input_ids"]
        inputs["labels"] = [[-100 for _ in x] for x in input_ids]

    def _insert_token_id_at_beginning_of_inputs(self, inputs: BatchEncoding, inserted_token_id: int):
        input_ids: list[list[int]] = inputs["input_ids"]
        attention_mask: list[list[int]] = inputs["attention_mask"]
        for i in range(len(input_ids)):
            input_ids[i].insert(0, inserted_token_id)
            attention_mask[i].insert(0, 1)
    
    

tokenizer: IndoNLGTokenizerForChat = IndoNLGTokenizerForChat.from_pretrained("indobenchmark/indogpt")
inputs = tokenizer.prepare_inputs_for_training(
    instructions=[
        "kamu adalah asisten yang dapat menjawab pertanyaan pengguna layaknya menjelaskan kepada orang yang berusia lima tahun.",
        "kamu adalah asisten yang berguna.",
    ],
    user_chats=[
        "dalam sepak bola apa gunanya menyia-nyiakan dua permainan pertama dengan terburu-buru - di tengah - bukan permainan terburu-buru biasa saya mendapatkannya",
        "halo",
    ],
    assistant_chats=[
        "jaga pertahanan tetap jujur, rasakan operan terburu-buru, buka permainan yang lewat. pelanggaran yang terlalu satu dimensi akan gagal. dan mereka yang bergegas ke tengah kadang-kadang dapat dibuka lebar-lebar untuk ukuran yard yang besar",
        "hai",
    ]
)
print(tokenizer.batch_decode(inputs["input_ids"]))
