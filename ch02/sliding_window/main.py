from gpt_data_loader import create_dataloader_v1


def main():
    with open("../the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)

if __name__ == "__main__":
    main()
