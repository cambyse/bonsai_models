"""Generate text from the Transformer trained by 01_train.py.

Run 01_train.py first to produce checkpoint.pt.
"""

import importlib.util
import pathlib

import torch

_TRAIN_MODULE_PATH = pathlib.Path(__file__).parent / "01_train.py"
_spec = importlib.util.spec_from_file_location("bonsai_train", _TRAIN_MODULE_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)  # runs everything in 01_train.py except the `train()` call


def main() -> None:
    if not train.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {train.CHECKPOINT_PATH}. Run 01_train.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train.MinimalTransformer().to(device)
    model.load_state_dict(torch.load(train.CHECKPOINT_PATH, map_location=device))
    model.eval()

    known_characters = "".join(train.characters)
    print("Type a prompt and press Enter to generate a continuation.")
    print("Empty line or Ctrl+D to quit.")
    print(f"Known characters ({len(known_characters)}): {known_characters!r}\n")

    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            break

        if not prompt:
            break

        unknown_characters = sorted(set(prompt) - set(train.char_to_id))
        if unknown_characters:
            print(f"  (skipped: unsupported characters {unknown_characters!r})\n")
            continue

        prompt_tokens = torch.tensor([train.encode(prompt)], dtype=torch.long, device=device)
        generated_tokens = model.generate(prompt_tokens, number_of_tokens=120)

        print(train.decode(generated_tokens[0].tolist()))
        print()


if __name__ == "__main__":
    main()
