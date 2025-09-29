from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

def run_chatbot():
    # Create ChatBot instance
    chatbot = ChatBot(
        'TerminalBot',
        logic_adapters=[
            'chatterbot.logic.BestMatch',
            'chatterbot.logic.MathematicalEvaluation'
        ],
        read_only=False
    )

    # Train chatbot with English corpus
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train("chatterbot.corpus.english")

    print("Type 'quit' to exit")
    while True:
        try:
            user_input = input("user: ")
            if user_input.lower() == "quit":
                print("bot: Goodbye!")
                break
            bot_response = chatbot.get_response(user_input)
            print(f"bot: {bot_response}")
        except (KeyboardInterrupt, EOFError):
            print("\nbot: Goodbye!")
            break

if __name__ == "__main__":
    run_chatbot()
