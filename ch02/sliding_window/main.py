import torch
from gpt_data_loader import create_dataloader_v1


def main():
    torch.manual_seed(123)
    with open("../the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab_size = 50257
    output_dim = 256
    # token embedding layer can represent each token of the vocabulary
    # by 256 parameters
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    # encode input text with snippets of 4 tokens
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    # create embedding layer with a batch size of 256
    # (every token is represented by 256 params)
    # initiatlize weights from inputs
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    context_length = max_length
    # add positional parameter to take position the token into account
    # The positional parameter reflects the position in the text
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # layer is initialized by the position as an additional parameter
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    # add token embeddings and positional params
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)


if __name__ == "__main__":
    main()
