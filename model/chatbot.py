import torch
import argparse
import sys
from pathlib import Path

from model.book_lookup import BookSession
from model.chat_data import CHAT_CHECKPOINT, load_chat_checkpoint
from model.chat_session import ChatSession, DEFAULT_MAX_TOKENS
from model.first_reply import FIRST_REPLY_CHECKPOINT, load_first_reply_classifier
from model.model import (
    get_device, load_completion_model, model_from_state, MODEL_PATH,
)

# CLI arguments
parser = argparse.ArgumentParser(description='Chat with the trained GPT language model')
parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                    help=f'Number of tokens to generate per prompt (default: {DEFAULT_MAX_TOKENS})')
modes = parser.add_mutually_exclusive_group()
modes.add_argument('--chat', nargs='?', const=CHAT_CHECKPOINT, type=Path, metavar='CHECKPOINT',
                   help='Use an Oz chat checkpoint (default path: model/model-chat-retrained.pth)')
modes.add_argument('--book', action='store_true', help='Look up answers in the bundled Oz book')
modes.add_argument('--checkpoint', type=Path, help='Use a different base checkpoint for text completion')
parser.add_argument('--first-reply-classifier', nargs='?', const=FIRST_REPLY_CHECKPOINT, type=Path,
                    metavar='CHECKPOINT', help='Classify the first reply to known facts in --chat mode')
parser.add_argument('--book-assist', action='store_true', help='Use cited answers from the Oz book in --chat mode')
args = parser.parse_args()
if args.chat and args.max_tokens < 1:
    parser.error('--max_tokens must be positive in chat mode.')
if args.first_reply_classifier and not args.chat:
    parser.error('--first-reply-classifier requires --chat.')
if args.book_assist and not args.chat:
    parser.error('--book-assist requires --chat.')

if args.book:
    print('Loading the Oz book and its answer reader...')
    try:
        session = BookSession.from_book()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Book lookup error: {error}\n')
    device = 'cpu'
else:
    device = get_device(allow_mps=args.chat is not None)
    checkpoint = args.chat or args.checkpoint or Path(MODEL_PATH)
    train_hint = (
        'Train a chat model with python -m model.chat_training.' if args.chat
        else 'Train with python -m model.training -batch_size 32 and pass the output path to --checkpoint.'
    )
    if not checkpoint.is_file():
        print(f"ERROR: No model file found at {checkpoint}")
        print(train_hint)
        sys.exit(1)

    print("Loading model parameters...")
    try:
        classifier = load_first_reply_classifier(args.first_reply_classifier) if args.first_reply_classifier else None
        if args.chat:
            state, tokenizer = load_chat_checkpoint(checkpoint)
            model = model_from_state(state, len(tokenizer.characters))
        else:
            model, tokenizer = load_completion_model(checkpoint)
        model.to(device)
    except Exception as e:
        print(f"ERROR: Failed to load model — {e}")
        print(train_hint)
        sys.exit(1)

    model.eval()
    if args.chat:
        book_session = None
        if args.book_assist:
            print('Loading the Oz book and its answer reader...')
            try:
                book_session = BookSession.from_book()
            except (OSError, ValueError, RuntimeError) as error:
                parser.exit(1, f'Book lookup error: {error}\n')
        session = ChatSession(model, tokenizer, device=device, max_tokens=args.max_tokens,
                              first_reply_classifier=classifier, book_session=book_session)
    print("Model loaded successfully!\n")

# Interactive chat loop
print("=" * 50)
print("  Oz Book Lookup" if args.book else "  GPT Language Model — Interactive Chat")
print("=" * 50)
print(f"  Device : {device}")
if not args.book:
    print(f"  Tokens : {args.max_tokens} per response")
if args.chat or args.book:
    print("  Type '/reset' to start a new conversation.")
print("  Type 'quit' or 'exit' to stop.")
print("=" * 50)
print()

while True:
    try:
        prompt = input("Prompt:\n")
        if prompt.lower().strip() in ('quit', 'exit'):
            print("Goodbye!")
            break
        if not prompt.strip():
            print("(Empty prompt — please enter text)\n")
            continue

        if args.chat or args.book:
            if prompt.strip().lower() == '/reset':
                session.reset()
                print("(Started a new conversation.)")
                continue
            try:
                reply = session.reply(prompt)
            except ValueError as error:
                print(f"({error})")
                continue
            print(f"\nAssistant: {reply.text}")
            if reply.source:
                source = reply.source
                print(f'Source: {source.section}, lines {source.start_line}–{source.end_line}')
            elif args.chat and reply.reply_method == 'classifier':
                print('Reply method: trained classifier')
            elif args.book_assist and reply.reply_method == 'abstain':
                print('Reply method: abstained')
            elif args.book_assist:
                print('Reply method: trained generator')
            print()
            continue

        context = torch.tensor(tokenizer.encode(prompt, ignore_unknown=True), dtype=torch.long, device=device)
        if context.numel() == 0:
            print("(No recognizable characters in prompt — "
                  "try using letters found in the training text.)\n")
            continue

        with torch.no_grad():
            generated = model.generate(context.unsqueeze(0),
                                       max_new_tokens=args.max_tokens)
        output_text = tokenizer.decode(generated[0].tolist())
        print(f"\nCompletion:\n{output_text}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break
