import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache_resource
def load_model_and_tokenizer(model_path: str = "ChatRec_Model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception:
        pass
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_reply(model, tokenizer, prompt: str, device: str, max_length: int = 80):
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


def main():
    st.title("Chat Reply Recommendation")
    model, tokenizer, device = load_model_and_tokenizer()

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:")
    if st.button("Send") and user_input:
        st.session_state.history.append(f"User: {user_input}")
        prompt = "\n".join(st.session_state.history[-6:]) + "\nAssistant:"
        reply = generate_reply(model, tokenizer, prompt, device)
        assistant_reply = reply.split("Assistant:")[-1].strip()
        st.session_state.history.append(f"Assistant: {assistant_reply}")

    # Display chat history
    for i, msg in enumerate(st.session_state.history[::-1]):
        if msg.startswith("Assistant:"):
            st.info(msg)
        else:
            st.write(msg)

if __name__ == "__main__":
    main()
