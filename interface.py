import argparse
import sys
from model_loader import load_pipeline
from chat_memory import SlidingWindowMemory


def chat_loop(model_name: str, window: int, max_new_tokens: int, temperature: float, top_p: float, use_gpu: bool):
    gen, task, tokenizer = load_pipeline(model_name=model_name, use_gpu=use_gpu)

    memory = SlidingWindowMemory(max_turns=window)

    print("👋 Hello! I'm your local CLI chatbot.")
    print("Type anything to chat. Commands: /exit, /reset, /help")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot. Goodbye!")
            break

        if not user_text:
            continue

        # Commands
        if user_text.lower() == "/exit":
            print("Exiting chatbot. Goodbye!")
            break
        if user_text.lower() == "/reset":
            memory.clear()
            print("Memory cleared!")
            continue
        if user_text.lower() == "/help":
            print("Commands: /exit (quit), /reset (clear memory), /help (this message)")
            continue

        # Build prompt and generate
        memory.add_user(user_text)
        prompt = memory.build_prompt(user_text)

        if task == "text-generation":
            outputs = gen(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_full_text=False,  # only the new text after the prompt
            )
            bot_reply = outputs[0]["generated_text"].strip()
            # Try to cut off at an extra user tag if the model keeps talking
            if "User:" in bot_reply:
                bot_reply = bot_reply.split("User:")[0].strip()
        else:  # text2text-generation
            outputs = gen(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            bot_reply = outputs[0]["generated_text"].strip()

        memory.add_bot(bot_reply)
        print(f"Bot: {bot_reply}")


def main():
    parser = argparse.ArgumentParser(description="Tiny local CLI chatbot using Hugging Face pipelines.")
    parser.add_argument("--model", type=str, default="google/flan-t5-large", help="HF model name (default: google/flan-t5-large)")
    parser.add_argument("--window", type=int, default=4, help="How many recent turns to remember (default: 4)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling (default: 0.95)")
    parser.add_argument("--use-gpu", action="store_true", help="Try to use GPU (if available)")

    args = parser.parse_args()
    try:
        chat_loop(
            model_name=args.model,
            window=args.window,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_gpu=args.use_gpu,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()