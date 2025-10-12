import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_path: str = "ChatRec_Model.pt", device: str = None):
    """Load the fine-tuned DistilGPT-2 model and tokenizer.

    Returns:
        model, tokenizer, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    # Load user-provided weights if available
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {model_path}")
    except Exception as e:
        print(f"Could not load weights from {model_path}: {e}\nUsing base distilgpt2 weights instead.")

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_reply(model, tokenizer, prompt: str, device: str, max_length: int = 50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def chat_loop(model, tokenizer, device):
    print("Start chatting (type 'exit' or 'quit' to stop)")
    history = []
    while True:
        user = input("You: ")
        if user.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Build prompt with short history
        history.append(f"User: {user}")
        prompt = "\n".join(history[-6:]) + "\nAssistant:"

        reply = generate_reply(model, tokenizer, prompt, device)
        # Post-process: keep only assistant reply portion
        assistant_reply = reply.split("Assistant:")[-1].strip()
        print("Bot:", assistant_reply)
        history.append(f"Assistant: {assistant_reply}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ChatRec_Model.pt", help="Path to model weights")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
