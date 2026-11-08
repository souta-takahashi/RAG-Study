from ollama import chat

print("AIに質問しています...")

response = chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": "RAGとは何か、初心者にもわかるように説明してください。"
        }
    ]
)

print("\n===== AIの回答 =====")
print(response.message.content)

print("\n===== 処理完了 =====")