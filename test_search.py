import chromadb
import ollama

# ① 保存済みデータベースを開く
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="security_study"
)

# ② ユーザーから質問を受け取る
question = input("質問を入力してください：")

# ③ 質問をベクトル化する
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

query_vector = response["embeddings"][0]

# ④ ベクトル検索を実行する
results = collection.query(
    query_embeddings=[query_vector],
    n_results=3
)

# ⑤ 検索結果を表示する
print("\n===== 検索結果 =====")

for i, document in enumerate(results["documents"][0]):

    distance = results["distances"][0][i]

    print(f"\n候補{i + 1}")
    print("文章：", document)
    print("距離：", distance)

print("\n===== 検索完了 =====")