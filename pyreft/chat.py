import torch
import transformers
import pyreft

# Define the prompt template (make sure it matches what was used during training)
prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

%s [/INST]
"""

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path where the fine-tuned model and tokenizer were saved
save_path = "./trained_llama2_model"

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(save_path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

# Load the fine-tuned ReFT model.
# (Assuming that pyreft's ReftModel subclass implements a from_pretrained method similar to transformers)
reft_model = pyreft.ReftModel.from_pretrained(save_path)
reft_model.set_device(device)

# Define your instruction
instruction = "Which dog breed do people think is cuter, poodle or doodle?"

# Prepare the prompt by inserting the instruction into the template
prompt = prompt_no_input_template % instruction

# Tokenize and move inputs to the appropriate device
prompt = tokenizer(prompt, return_tensors="pt").to(device)

# Identify the last token position for the intervention
base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position

# Generate a response using the fine-tuned model
_, reft_response = reft_model.generate(
    prompt,
    unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True,
    max_new_tokens=512,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True
)

# Decode and print the response
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

"""
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Which dog breed do people think is cuter, poodle or doodle? [/INST]
🐶🔢💬🍁
"""