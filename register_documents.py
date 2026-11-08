import chromadb
import ollama
from pathlib import Path
from pypdf import PdfReader

# ① フォルダとデータベースの設定
DOCUMENT_DIR = Path("./documents")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="security_study"
)

# ② 文章を小分けにする関数
def split_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ③ ファイルから文章を読み込む
all_chunks = []
all_ids = []
all_metadata = []

for file_path in DOCUMENT_DIR.iterdir():

    extracted_pages = []

    # PDFの場合
    if file_path.suffix.lower() == ".pdf":

        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            extracted_pages.append((page_number, text))

    # TXTの場合
    elif file_path.suffix.lower() == ".txt":

        text = file_path.read_text(encoding="utf-8")

        extracted_pages.append((0, text))

    else:
        continue

    # ④ 読み込んだ文章をチャンクに分割
    for page_number, text in extracted_pages:

        chunks = split_text(text)

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            all_ids.append(
                f"{file_path.name}-p{page_number}-c{chunk_number}"
            )

            all_metadata.append({
                "source": file_path.name,
                "page": page_number,
                "chunk": chunk_number
            })


# ⑤ ベクトル化してデータベースに登録
if all_chunks:

    response = ollama.embed(
        model="nomic-embed-text",
        input=all_chunks
    )

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=response["embeddings"],
        metadatas=all_metadata
    )

    print("資料の登録が完了しました！")
    print("今回の登録・更新件数：", len(all_chunks))

else:
    print("読み込める文章が見つかりませんでした。")

print("データベースの総登録件数：", collection.count())