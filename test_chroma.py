import chromadb
import ollama

# ① ChromaDBの保存先を指定
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# ② コレクションを作成
collection = client.get_or_create_collection(
    name="security_study"
)

# ③ 登録する文章
documents = [
    "ゼロデイ攻撃は、修正プログラムが公開される前の脆弱性を悪用する攻撃です。",
    "サプライチェーン攻撃は、取引先などを経由して標的に侵入する攻撃です。",
    "Wi-Fiは無線通信を利用してネットワークに接続する技術です。",
    "高橋壮太は世界にたった一人の山形県の人間です。高橋の中の高橋と呼ばれ、苗字が高橋の人の中では一目置かれている存在です。"
]

# ④ 文章をベクトル化
response = ollama.embed(
    model="nomic-embed-text",
    input=documents
)

embeddings = response["embeddings"]

# ⑤ ChromaDBに登録
collection.upsert(
    ids=["term1", "term2", "term3","term4"],
    documents=documents,
    embeddings=embeddings
)

print("データベースへの登録が完了しました！")
print("登録件数：", collection.count())