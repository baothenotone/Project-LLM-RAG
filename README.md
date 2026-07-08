# Project_LLM

He thong hoi-dap ho tro tra cuu quy che dao tao tieng Viet su dung RAG co trich dan nguon.

## Cau truc thu muc

```text
Project_LLM/
|-- data/
|   |-- raw/              Tai lieu quy che goc
|   |-- processed/        Van ban da trich xuat va lam sach
|   `-- chunks/           Cac doan van ban kem metadata nguon
|-- vector_store/         FAISS/Chroma index va metadata
|-- src/                  Ma nguon pipeline RAG
|-- notebooks/            Notebook thu nghiem, tao moi khi can
|-- tests/                Kiem thu cac module chinh
|-- requirements.txt
`-- README.md
```

## Pham vi

Du an tap trung vao RAG co ban: doc tai lieu, chia chunk, tao embedding, truy xuat doan lien quan, sinh cau tra loi va hien thi nguon trich dan.
