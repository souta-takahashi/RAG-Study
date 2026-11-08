
# RAG-Study

## 概要
PythonとOllamaを使用した個人学習用RAGシステムです。

PDFやTXTファイルを読み込み、
ベクトルデータベースを利用して関連情報を検索し、
AIが資料を参考に回答する仕組みを実装しています。

## 使用技術
- Python 3.14
- Ollama
- Qwen2.5:7b
- Nomic Embed Text
- ChromaDB
- pypdf

## 主な機能
- PDF・TXTファイルの読み込み
- 文章のチャンク分割
- Embeddingによるベクトル化
- ChromaDBへのデータ登録
- ベクトル検索
- AIによる回答生成
- 参考資料・ページ番号の表示

## 実行方法

仮想環境を有効化：

```cmd
.venv\Scripts\activate
```

資料の登録：

```cmd
python register_documents.py
```

AIへの質問：

```cmd
python rag_chat.py
```

## 学習目的
RAGの仕組みを理解し、個人で実装することを目的としています。
