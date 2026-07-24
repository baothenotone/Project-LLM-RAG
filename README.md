# Hệ thống hỏi–đáp quy chế đào tạo HUSC sử dụng RAG

Đây là **prototype hệ thống hỏi–đáp hỗ trợ tra cứu quy chế đào tạo tiếng Việt** của Trường Đại học Khoa học, Đại học Huế (HUSC), được xây dựng theo kiến trúc **Retrieval-Augmented Generation (RAG)** và có hiển thị thông tin nguồn tham khảo.

Hệ thống tiếp nhận câu hỏi của người dùng, truy xuất các đoạn văn bản liên quan trong kho tài liệu quy chế, sau đó cung cấp ngữ cảnh cho mô hình Gemini để tạo câu trả lời bằng tiếng Việt.

## Mục tiêu

- Hỗ trợ sinh viên tra cứu nhanh các quy định và thông tin học vụ.
- Giảm thời gian tìm kiếm thủ công trong các tài liệu PDF dài.
- Tạo câu trả lời dựa trên nội dung tài liệu đã truy xuất.
- Hiển thị tên tài liệu và số trang để người dùng kiểm tra lại nguồn.
- Xây dựng một prototype RAG có cấu trúc rõ ràng, phù hợp với phạm vi đề tài 2 tín chỉ.

## Chức năng hiện tại

- Đọc các tài liệu PDF trong `data/raw/`.
- Trích xuất và làm sạch văn bản bằng Docling.
- Chia tài liệu thành các `chunk` có nội dung và metadata nguồn.
- Tạo embedding tiếng Việt cho từng chunk.
- Lưu embedding dưới dạng ma trận NumPy.
- Tạo embedding cho câu hỏi người dùng.
- Tính độ tương đồng và truy xuất `top-k` chunk liên quan.
- Gửi câu hỏi cùng ngữ cảnh truy xuất đến Gemini.
- Sinh câu trả lời có đánh dấu `[Nguồn 1]`, `[Nguồn 2]`, ...
- Chạy thử toàn bộ hệ thống trên giao diện dòng lệnh.

> Giao diện Streamlit trong `app.py` đang được phát triển và chưa phải thành phần chính của phiên bản hiện tại.

## Kiến trúc hệ thống

```text
Tài liệu PDF
    │
    ▼
preprocess.py
    │
    ├── Trích xuất văn bản
    ├── Làm sạch dữ liệu
    ├── Chia chunk
    └── Gắn metadata nguồn
    │
    ▼
data/chunks/chunks.json
    │
    ▼
embeddings.py
    │
    ├── Tạo document embeddings
    ├── Lưu embeddings.npy
    └── Lưu metadata.json
    │
    ▼
retriever.py
    │
    ├── Tạo query embedding
    ├── Tính cosine similarity
    └── Chọn top-k chunk
    │
    ▼
rag_pipeline.py
    │
    ├── Xây dựng context
    ├── Gọi Gemini API
    └── Tạo câu trả lời
    │
    ▼
Câu trả lời kèm nguồn tham khảo
```

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Xử lý PDF | Docling |
| Chunking | Chunking theo cấu trúc và giới hạn token |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Kích thước embedding | 384 chiều |
| Lưu trữ vector | NumPy (`embeddings.npy`) |
| Độ tương đồng | Cosine similarity trên vector đã chuẩn hóa |
| Large Language Model | Google Gemini |
| Quản lý biến môi trường | `python-dotenv` |
| Giao diện dự kiến | Streamlit |

Phiên bản hiện tại sử dụng ma trận NumPy để lưu và tìm kiếm embedding, chưa phụ thuộc vào FAISS hoặc một vector database riêng.

## Cấu trúc thư mục

```text
Project_LLM/
├── data/
│   ├── raw/
│   │   └── *.pdf
│   ├── extracted/
│   ├── chunks/
│   │   ├── chunks.json
│   │   └── rejected_chunks.json
│   ├── reports/
│   │   ├── preprocess_report.json
│   │   └── chunks_preview.txt
│   └── document_catalog.json
│
├── vector_store/
│   ├── embeddings.npy
│   ├── metadata.json
│   ├── embeddings_preview.csv
│   └── skipped_chunks.json
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   ├── citations.py
│   ├── app.py
│   └── test.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

Một số file báo cáo hoặc file kiểm tra chỉ xuất hiện sau khi chạy pipeline.

## Vai trò của các file trong `src/`

### `preprocess.py`

Tiền xử lý toàn bộ tài liệu PDF:

```text
PDF
→ trích xuất văn bản
→ làm sạch
→ chia chunk
→ gắn metadata
→ chunks.json
```

Metadata của mỗi chunk có thể gồm:

- `chunk_id`
- `source_file`
- `document_title`
- `pages`
- `headings`
- `article`
- `clause`
- `content`
- `embedding_text`

### `embeddings.py`

File này đã gộp chức năng của `embeddings.py` cũ và `ingest.py`.

Nhiệm vụ chính:

- Tải embedding model.
- Tạo embedding cho các chunk.
- Tạo embedding cho câu hỏi.
- Kiểm tra và bỏ qua chunk không hợp lệ.
- Lưu `embeddings.npy`.
- Lưu `metadata.json`.
- Tạo `embeddings_preview.csv` để kiểm tra.

Vì chức năng xây dựng vector store đã được gộp vào đây nên project hiện tại **không còn cần `src/ingest.py`**.

### `retriever.py`

Thực hiện quá trình Retrieval:

- Đọc `embeddings.npy`.
- Đọc `metadata.json`.
- Chuyển câu hỏi thành query embedding.
- Tính cosine similarity.
- Trả về các chunk có điểm tương đồng cao nhất.

### `rag_pipeline.py`

Điều phối toàn bộ luồng hỏi–đáp:

```text
Câu hỏi
→ Retriever
→ Top-k chunks
→ Context
→ Gemini
→ Câu trả lời
```

File này đọc các biến cấu hình Gemini từ `.env`.

### `citations.py`

Định dạng thông tin nguồn từ metadata, chẳng hạn:

```text
[Nguồn 1] Tên tài liệu, trang 12–13
```

### `test.py`

Dùng để chạy thử hệ thống hỏi–đáp trên terminal và kiểm tra kết quả đầu ra.

### `app.py`

Dành cho giao diện Streamlit. Thành phần này đang được tiếp tục hoàn thiện.

## Cài đặt

### 1. Tạo virtual environment

Trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Trên Linux hoặc macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Các dependency chính của phiên bản hiện tại gồm:

```text
docling
sentence-transformers
numpy
google-genai
python-dotenv
streamlit
```

Nếu `requirements.txt` chưa có `docling` hoặc `google-genai`, cần bổ sung hai package này trước khi chạy toàn bộ pipeline.

## Cấu hình Gemini API

Tạo file `.env` tại thư mục gốc của project:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=your_gemini_model
```

Không đưa API key thật vào source code và không commit file `.env` lên GitHub.

File `.gitignore` cần có:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

## Cách chạy

Các lệnh sau được thực hiện tại thư mục gốc của project.

### Bước 1: Chuẩn bị tài liệu

Đặt các file PDF vào:

```text
data/raw/
```

### Bước 2: Tiền xử lý tài liệu

```bash
python src/preprocess.py
```

Kết quả chính:

```text
data/chunks/chunks.json
data/reports/preprocess_report.json
data/reports/chunks_preview.txt
```

Nên kiểm tra `chunks_preview.txt` trước khi tạo embedding để phát hiện lỗi trích xuất PDF hoặc chunk bị cắt sai.

### Bước 3: Tạo vector store

```bash
python src/embeddings.py
```

Kết quả:

```text
vector_store/embeddings.npy
vector_store/metadata.json
vector_store/embeddings_preview.csv
vector_store/skipped_chunks.json
```

Lần chạy đầu tiên có thể mất thêm thời gian vì chương trình phải tải embedding model từ Hugging Face.

### Bước 4: Kiểm tra Retrieval

```bash
python src/retriever.py
```

Ví dụ câu hỏi:

```text
Sinh viên liên hệ đơn vị nào khi gặp vấn đề về học vụ?
```

Retriever sẽ hiển thị các chunk có điểm tương đồng cao nhất, kèm tên file và số trang.

### Bước 5: Chạy hệ thống hỏi–đáp

```bash
python src/rag_pipeline.py
```

Ví dụ:

```text
Nhập câu hỏi: Phòng Đào tạo đại học và Công tác sinh viên chịu trách nhiệm gì?
```

Để kết thúc chương trình:

```text
exit
```

### Bước 6: Chạy giao diện Streamlit

Khi `app.py` đã được hoàn thiện:

```bash
streamlit run src/app.py
```

## Luồng dữ liệu

```text
data/raw/*.pdf
    ↓
data/chunks/chunks.json
    ↓
vector_store/embeddings.npy
vector_store/metadata.json
    ↓
top-k chunks
    ↓
Gemini
    ↓
câu trả lời kèm nguồn
```

## Ví dụ đầu ra

```text
Câu trả lời:
Phòng Đào tạo đại học và Công tác sinh viên có trách nhiệm
phối hợp với các đơn vị liên quan trong việc tổ chức, quản lý
và kiểm tra hoạt động đào tạo...

Nguồn tham khảo:
[Nguồn 1] ten_tai_lieu.pdf, trang 12
[Nguồn 2] ten_tai_lieu_khac.pdf, trang 4–5
```

Nội dung thực tế phụ thuộc vào tài liệu trong `data/raw/` và các chunk được Retriever lựa chọn.

## Kiểm tra chất lượng dữ liệu

Trước khi đánh giá chất lượng câu trả lời, cần kiểm tra lần lượt:

1. Nội dung được trích xuất từ PDF có đúng không.
2. Chunk có giữ được câu hoặc ý nghĩa hoàn chỉnh không.
3. Metadata có đúng tên tài liệu và số trang không.
4. Retriever có trả về đúng tài liệu liên quan không.
5. Câu trả lời của Gemini có bám sát các nguồn đã truy xuất không.

Nếu Retrieval sai thì Generation thường cũng sẽ sai hoặc thiếu căn cứ.

## Giới hạn hiện tại

- Chất lượng đầu vào phụ thuộc vào text layer bên trong PDF.
- PDF scan hoặc PDF có bố cục phức tạp có thể cần OCR và xử lý riêng.
- Semantic search đôi khi trả về chunk có ý nghĩa gần nhưng không trả lời trực tiếp câu hỏi.
- Chưa có Hybrid Search giữa BM25 và vector search.
- Chưa có re-ranking model.
- Chưa có bộ câu hỏi đánh giá tự động hoàn chỉnh.
- Giao diện Streamlit chưa hoàn thiện.
- Hệ thống hiện là prototype phục vụ học tập và nghiên cứu, không thay thế văn bản quy định chính thức.

## Hướng phát triển

- Hoàn thiện giao diện Streamlit.
- Chuẩn hóa cách hiển thị trích dẫn.
- Thêm ngưỡng điểm Retrieval để loại kết quả kém liên quan.
- Xây dựng bộ câu hỏi kiểm thử có đáp án tham chiếu.
- Đánh giá các chỉ số Retrieval như `Hit@k`, `Recall@k` và `MRR`.
- Thử nghiệm Hybrid Search hoặc re-ranking khi cần cải thiện chất lượng.
- Bổ sung cơ chế từ chối trả lời khi tài liệu không đủ căn cứ.

## Lưu ý

Câu trả lời của hệ thống chỉ dùng để hỗ trợ tra cứu. Khi cần quyết định chính thức, người dùng phải đối chiếu lại tài liệu quy chế gốc và các văn bản đang còn hiệu lực.
