import re
import tiktoken
from importlib.metadata import version

print("tiktoken.version:", version("tiktoken"))


def main():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))

    preprocessed = re.split(r'([,.;:\?_!"\(\)\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("Preprocessed length:", len(preprocessed))

    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_words)}
    tokenizer = tiktoken.get_encoding("gpt2")
    text1 = "Hello, do you like, tea?"
    text2 = "In the sunlit terraces of the someunknownPlace."
    text = " <|endoftext|> ".join((text1, text2))
    print(f"Input: {text}")
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(ids)
    restored = tokenizer.decode(ids)
    print(restored)


if __name__ == "__main__":
    main()
