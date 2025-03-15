import torch
from huggingface_hub import login
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()
import os

# enable hf transfer for the huggingface hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# region configuration
# set model id, fine tune tag, and cache directory
model_id = "Qwen/Qwen1.5-7B-Chat"
fine_tune_tag = "faa-balloon-flying-handbook"
cache_dir = "cache"

# flag to determine if model should be uploaded to huggingface hub
upload_to_hf = True

# set up lora configuration and specify target modules for fine tuning
peft_config = LoraConfig(
    r=16,
    modules_to_save=["lm_head", "embed_tokens"],
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# system message used to instruct the model during fine tuning
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


# endregion

def load_model_and_tokenizer(model_id, cache_dir):
    # configure model loading parameters including device mapping, remote code trust, and torch dtype
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        # bf16=True,
        # tf32=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
        cache_dir=cache_dir,
    )
    # set quantization configuration for 4-bit loading using bitsandbytes
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    # load the causal language model from huggingface with the specified kwargs
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # load the tokenizer for the model and set pad token if not already defined
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_conversation(sample):
    # create a conversation dict with system, user, and assistant roles from the sample
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }


def main():
    # if upload flag is true, login to huggingface hub using the token from environment variables
    if upload_to_hf:
        login(
            token=os.getenv('HF_TOKEN'),  # add your token here
            add_to_git_credential=True
        )

    # load the model and tokenizer from huggingface with the specified configuration
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)

    # enable gradient checkpointing to reduce memory usage during training
    model.gradient_checkpointing_enable()
    # wrap the model with peft using the lora configuration for parameter-efficient fine tuning
    model = get_peft_model(model, peft_config)

    # load the training and validation datasets from the specified dataset id
    dataset_id = "gsantopaolo/faa-balloon-flying-handbook"
    train_dataset = load_dataset(dataset_id, split="train")
    validation_dataset = load_dataset(dataset_id, split="validation")

    # apply conversation template to each sample in the datasets and remove original columns
    train_dataset = train_dataset.map(create_conversation, remove_columns=train_dataset.features, batched=False)
    validation_dataset = validation_dataset.map(create_conversation, remove_columns=validation_dataset.features,
                                                batched=False)
    # limit the training and validation datasets to a subset of samples for faster experimentation
    # train_dataset = train_dataset.take(10)
    # validation_dataset = validation_dataset.take(1)

    def tokenize(sample):
        # for each sample, convert the list of message dictionaries into a single string per conversation
        conversation_strs = []
        for conversation in sample["messages"]:
            # join each message in the conversation with role and content
            conv_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
            conversation_strs.append(conv_str)
        # tokenize the conversation strings with truncation to a maximum length
        tokenized = tokenizer(conversation_strs, truncation=True, max_length=1024)
        # add the raw conversation text as a new field for the trainer
        tokenized["text"] = conversation_strs
        return tokenized

    # apply tokenization to the training dataset in batched mode
    tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
    # remove columns that are not required for training
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(
        [col for col in tokenized_train_dataset.column_names if col not in ["input_ids", "attention_mask", "text"]]
    )

    # apply tokenization to the validation dataset
    tokenized_validation_dataset = validation_dataset.map(tokenize, batched=True)

    # extract model name and dataset name for constructing the save directory path
    model_name = model_id.split("/")[-1]
    dataset_name = dataset_id.split("/")[-1]

    # set the context length for tokenization
    context_length = 1024

    # create a directory path to save training results and fine tuned model artifacts
    save_dir = f"./results/{model_name}-{fine_tune_tag}/"
    print("save directory:", save_dir)

    # region sft trainer
    # initialize the sft trainer with model, tokenizer, datasets, and training arguments
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        args=TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=4e-5,  # previously 1e-6,
            # lr_scheduler_type="cosine",
            lr_scheduler_type="constant",
            warmup_ratio=0.1,
            save_steps=50,
            bf16=True,

            # evaluation parameters
            per_device_eval_batch_size=2,
            # evaluation_strategy options: "steps" or "epoch"; set to "steps" to evaluate periodically
            evaluation_strategy="steps",
            do_eval=True,
            eval_steps=50,

            # logging parameters for tensorboard and debug output
            logging_strategy="steps",
            logging_steps=5,
            report_to=["tensorboard"],
            save_strategy="epoch",
            seed=42,
            output_dir=save_dir,
            log_level="debug",
        ),
    )
    # endregion

    # disable cache to ensure proper training behavior
    model.config.use_cache = False

    # start training the model
    trainer.train()

    # if the upload flag is true, upload the fine tuned model and adapters to huggingface hub
    if upload_to_hf:
        print("\nuploading model to HF...")
        adapter_model = f"gsantopaolo/{model_name}-{fine_tune_tag}"

        # save the model with separate adapter weights so that the adapter configuration is preserved
        model.save_pretrained(f"{save_dir}", push_to_hub=True, use_auth_token=True)
        model.push_to_hub(adapter_model, use_auth_token=True, max_shard_size="10gb", use_safetensors=True)
        # end upload to hf

    # save the adapter separately in a local directory
    adapter_checkpoint_dir = f"{save_dir}/adapters-local"
    model.save_pretrained(adapter_checkpoint_dir)
    tokenizer.save_pretrained(adapter_checkpoint_dir)
    print(f"\nadapter checkpoint saved to: {adapter_checkpoint_dir}")

    # merge the adapter with the base model to create a self-contained merged model
    merged_model = model.merge_and_unload()
    merged_checkpoint_dir = f"f{save_dir}/merged-local"
    merged_model.save_pretrained(merged_checkpoint_dir)
    tokenizer.save_pretrained(merged_checkpoint_dir)
    print(f"merged model saved to: {merged_checkpoint_dir}")


if __name__ == "__main__":
    main()
