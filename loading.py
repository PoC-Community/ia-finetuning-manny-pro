from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token (because the end of the sentence is not detected by the model)
tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model '{model_name}' loaded successfully!")
print(f"Model has {model.num_parameters():,} parameters")

# Test the model with a simple question
test_input = "What is the capital of France ?"
inputs = tokenizer.encode(test_input, return_tensors='pt')
outputs = model.generate(inputs, max_length=30)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n📝 Test question: {test_input}")
print(f"💬 Model response: {response}")

import json

with open("false_capital_data.json", "r") as file: 
        data = json.load(file)

print(f"Dataset loaded: {len(data)} examples")
print(f"First example: {data[0]}")