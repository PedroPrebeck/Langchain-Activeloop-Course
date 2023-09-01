from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")

print(f"Tokens: {tokenizer.convert_ids_to_tokens(token_ids)}")
print(f"Tokens IDs: {token_ids}")