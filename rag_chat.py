import chromadb
import ollama

# ① データベースを開く
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="security_study"
)

# ② 質問を受け取る
question = input("質問を入力してください：")

# ③ 質問をベクトル化
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

query_vector = response["embeddings"][0]

# ④ 関連資料を検索
results = collection.query(
    query_embeddings=[query_vector],
    n_results=min(3, collection.count()),
    include=["documents", "metadatas", "distances"]
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

# ⑤ 資料と出典情報をまとめる
context_parts = []

print("\n===== 検索された資料 =====")

for i, document in enumerate(documents):

    metadata = metadatas[i] or {}

    source = metadata.get("source", "不明")
    page = metadata.get("page", 0)

    if page > 0:
        location = f"{source}・{page}ページ"
    else:
        location = source

    context_parts.append(
        f"【参考資料{i + 1}：{location}】\n{document}"
    )

    print(f"\n参考資料{i + 1}")
    print("ファイル：", source)
    print("ページ：", page if page > 0 else "該当なし")
    print("文章：", document)

context = "\n\n".join(context_parts)

# ⑥ AIへの指示を作成
prompt = f"""
あなたは情報セキュリティの学習を支援するAIです。

以下の参考資料を根拠として質問に回答してください。

資料に記載されていない情報については、
資料からは判断できないと明示してください。

回答の最後に、参考にした資料名とページ番号を記載してください。

【参考資料】
{context}

【質問】
{question}
"""

# ⑦ AIに回答させる
response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# ⑧ 回答を表示
print("\n===== RAGの回答 =====")
print(response.message.content)

# ⑨ 出典を表示
print("\n===== 参照元 =====")

for i, metadata in enumerate(metadatas):

    metadata = metadata or {}

    print(
        f"{i + 1}. {metadata.get('source', '不明')}"
        f" / ページ: {metadata.get('page', '不明')}"
    )