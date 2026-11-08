import chromadb
import ollama

# ① データベースを開く
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="security_study"
)

# ② 質問を入力
question = input("検索したい内容を入力してください：")

# ③ 質問をベクトル化
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

query_vector = response["embeddings"][0]

# ④ ChromaDBで検索
result_count = min(5, collection.count())

if result_count == 0:
    print("データベースに資料がありません。")
    raise SystemExit

results = collection.query(
    query_embeddings=[query_vector],
    n_results=result_count,
    include=["documents", "metadatas", "distances"]
)

# ⑤ 検索結果を表示
print("\n===== 検索結果 =====")

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, document in enumerate(documents):

    metadata = metadatas[i] or {}

    print(f"\n--- 候補 {i + 1} ---")

    print("文章：", document)

    print("参照ファイル：",
          metadata.get("source", "不明"))

    page = metadata.get("page", 0)

    if page > 0:
        print("ページ：", page)
    else:
        print("ページ：テキストファイルのため該当なし")

    print("チャンク番号：",
          metadata.get("chunk", "不明"))

    print("距離：", distances[i])

print("\n===== 検索完了 =====")