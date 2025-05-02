import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("token size: ", encoder.n_vocab)

text = "Hello, world!"
tokens = encoder.encode(text)
print(f"tokens for {text}: {tokens}")

decoded_text = encoder.decode(tokens)
print(f"decoded text for {tokens}: {decoded_text}")
