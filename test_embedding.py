import ollama

# ベクトル化したい文章
documents = [
    "ゼロデイ攻撃は、修正プログラムが公開される前の脆弱性を悪用する攻撃です。",
    "サプライチェーン攻撃は、取引先などを経由して標的に侵入する攻撃です。",
    "Wi-Fiは無線通信を利用してネットワークに接続する技術です。",
    "高橋壮太は世界にたった一人の山形県の人間です。高橋の中の高橋と呼ばれ、苗字が高橋の人の中では一目置かれている存在です。"
]

print("文章をベクトル化しています...")

for i, document in enumerate(documents):

    response = ollama.embed(
        model="nomic-embed-text",
        input=document
    )

    vector = response["embeddings"][0]

    print(f"\n===== 文章 {i + 1} =====")
    print(document)

    print("\nベクトルの次元数：")
    print(len(vector))

    print("\nベクトルの先頭10個：")
    print(vector[:10])

print("\nすべて完了しました！")