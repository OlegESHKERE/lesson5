import os
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate

load_dotenv()

class SimpleChatBot:
    def __init__(self, memory_limit=5):
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        self.llm = OpenRouter(
            model="openai/gpt-4o-mini",
            api_key=self.api_key,
            max_tokens=1024,
            context_window=4096,
            temperature=0.7
        )
        
        os.environ["OPENROUTER_HTTP_REFERER"] = "https://your-app-url.com"
        os.environ["OPENROUTER_TITLE"] = "Ukrainian ChatBot"
        
        self.memory = ChatMemoryBuffer(token_limit=memory_limit * 100)
        self.prompt_template = PromptTemplate(
            "Історія розмови:\n{history}\n\nКористувач: {query}\n\nАсистент:"
        )

    def chat(self):
        print("Привіт! Я твій AI-асистент. Пиши свої повідомлення (або 'вихід' щоб завершити).")
        while True:
            user_input = input("Ти: ")
            if user_input.strip().lower() in ['вихід', 'exit', 'quit']:
                print("До нових зустрічей!")
                break
            self.handle_input(user_input)

    def handle_input(self, user_input):
        self.memory.put(f"Користувач: {user_input}")
        prompt = self.prompt_template.format(
            history=self.memory.get_all(),
            query=user_input
        )
        
        try:
            response = self.llm.complete(prompt).text
            print(f"Асистент: {response}")
            self.memory.put(f"Асистент: {response}")
        except Exception as e:
            print("Сталася помилка під час обробки запиту:", e)
            if "context length" in str(e):
                self.memory.reset()
                print("Пам'ять очищена через перевищення обмеження довжини.")

if __name__ == "__main__":
    bot = SimpleChatBot()
    bot.chat()