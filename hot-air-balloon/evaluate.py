import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

system_message = """answer the given balloon flying handbook question by providing a clear, detailed explanation that references guidance from the balloon flying handbook, operational procedures, and relevant flight concepts.

provide a detailed breakdown of your answer, beginning with an explanation of the question and its context within the balloon flying handbook, followed by step-by-step reasoning based on the information provided in the handbook and applicable flight operation procedures. use logical steps that build upon one another to arrive at a comprehensive solution.

# steps

1. **understand the question**: restate the given question and clearly identify the main query along with any relevant details about balloon operations, safety procedures, or flight scenarios as discussed in the balloon flying handbook.
2. **handbook context**: explain the relevant procedures and guidelines as outlined in the balloon flying handbook. reference specific sections of the handbook, such as pre-flight checks, flight planning, emergency procedures, and operational parameters central to the question.
3. **detailed explanation**: provide a step-by-step breakdown of your answer. describe how you arrived at each conclusion by citing pertinent sections of the handbook and relevant operational standards.
4. **double check**: verify that your explanation is consistent with the guidelines in the balloon flying handbook and accurate according to current practices. mention any alternative methods or considerations if applicable.
5. **final answer**: summarize your answer clearly and concisely, ensuring that it is accurate and fully addresses the question.

# notes

- clearly define any terms or procedures specific to balloon flight operations as described in the handbook.
- include relevant procedural steps, operational parameters, or safety guidelines where applicable to support your answer.
- assume a familiarity with basic flight operation concepts while avoiding overly technical jargon unless it is commonly used in the ballooning community.
"""

def load_model_and_tokenizer(model_id, cache_dir="cache"):
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
        cache_dir=cache_dir,
    )
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def stream(model, user_prompt, model_type, tokenizer, checkpoint=""):
    if model_type == "base":
        eval_model = model
    elif model_type == "fine-tuned":
        # Load the fine-tuned adapter weights on top of the base model
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cuda")
    else:
        print("You must set the model_type to 'base' or 'fine-tuned'")
        exit()

    eval_model.config.use_cache = True
    # Include both the system message and user message in the conversation.
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt.strip()}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([inputs], return_tensors="pt", add_special_tokens=False).to("cuda")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    streamer = TextStreamer(tokenizer)
    print(f"Model device: {next(eval_model.parameters()).device}")
    print(f"Input IDs device: {inputs['input_ids'].device}")

    _ = eval_model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.cuda.empty_cache()
    gc.collect()

def evaluation(model, model_type, tokenizer, checkpoint=""):
    questions = [
        "In the context of hot air balloon, What should a pilot establish during the initial practice stages of performing a descent maneuver??",
        "In the context of hot air balloon, What are the consequences of flying a balloon without an up-to-date annual inspection??",
        "In the context of hot air balloon, What can be inferred about the effectiveness of air as a conductor of heat based on the text?",
    ]
    for question in questions:
        print("\n" + "=" * 50)
        print("User Question:", question)
        stream(model, question, model_type, tokenizer, checkpoint)
        print("\n" + "=" * 50 + "\n")

def main():
    model_id = "Qwen/Qwen1.5-7B-Chat"
    model, tokenizer = load_model_and_tokenizer(model_id)

    print("Evaluating the Base Model:")
    evaluation(model, "base", tokenizer)

    # Specify the directory or identifier where your fine-tuned adapter checkpoint is stored.
    ft_checkpoint = "results/Qwen1.5-7B-Chat-faa-balloon-flying-handbook/"
    print("Evaluating the Fine-Tuned Model:")
    # evaluation(model, "fine-tuned", tokenizer, checkpoint=ft_checkpoint)
    evaluation(model, "base", tokenizer)

if __name__ == "__main__":
    main()
