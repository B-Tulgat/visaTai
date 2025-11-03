# visaTai

A RAG-based information system for Irish immigration queries, specifically designed for Mongolian citizens. The system scrapes official Irish immigration sources, processes the data through semantic clustering, and delivers answers in Mongolian using retrieval-augmented generation.

**Disclaimer**: This readme is made with the help of AI LLM model.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Data Sources](#data-sources)

---

## Overview

visaTai automates the collection and processing of Irish immigration information from official sources. It provides accurate, context-aware answers about:

- Student and tourist visa requirements
- Living expenses in Ireland
- VFS application procedures for Mongolian citizens
- PPS number information
- Residency permits
- Educational institutions and language courses

The system converts queries into Mongolian responses, making Irish immigration information accessible to Mongolian speakers.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Collection                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Irish Immig. │  │   Numbeo     │  │     VFS      │           │
│  │   Website    │  │  (Expenses)  │  │   Mongolia   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                │                   │
│         └──────────────────┴────────────────┘                   │
│                            │                                    │
                             ▼                                    │
│                    ┌────────────────┐                           │
│                    │  PDF Download  │                           │
│                    │   & OCR Scan   │                           │
│                    └───────┬────────┘                           │
│                            │                                    │
│                    ┌───────▼────────┐                           │
│                    │   data.txt     │                           │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Data Processing                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Tokenization │→ │  Embeddings  │→ │   DBSCAN     │           │
│  │    (NLTK)    │  │(Transformers)│  │  Clustering  │           │
│  └──────────────┘  └──────────────┘  └──────┬───────┘           │
│                                             │                   │
│                                             ▼                   │
│                                      ┌────────────────┐         │
│                                      │  curated.txt   │         │
│                                      └────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Query     │→ │   Retrieve   │→ │  Llama 3.2   │           │
│  │  (Any Lang)  │  │   Top-K=20   │  │  Generation  │           │
│  └──────────────┘  └──────────────┘  └──────┬───────┘           │
│                                              │                  │
│                                      ┌───────▼────────┐         │
│                                      │  Translation   │         │
│                                      │  to Mongolian  │         │
│                                      └───────┬────────┘         │
│                                              │                  │
│                                      ┌───────▼────────┐         │
│                                      │     Answer     │         │
│                                      └────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Data Collection

The system scrapes and processes data from:

- Irish Immigration Service website (HTML extraction)
- Numbeo living expenses (tabularized as Parquet)
- VFS Global Mongolia-to-Ireland requirements
- Educational institutions:
  - Language courses: Erin School, IBAT College, Griffith College
  - Universities: Trinity College Dublin, University College Cork, Griffith College

PDF files are automatically downloaded and OCR scanned to text. All data is consolidated into `data/data.txt`.

### 2. Data Curation

Raw text undergoes semantic processing:



**Process:**
1. Sentence tokenization using NLTK Punkt
2. Embedding generation via SentenceTransformer
3. Cosine similarity computation between embeddings
4. DBSCAN clustering to group semantically similar content
5. Concatenation of related sentences into cohesive knowledge points
6. Output to `curated.txt`

### 3. RAG Pipeline

The curated data is converted to a vector database and queried:

```python
def rag_pipeline(query):
    docs = retrieve_similar_documents(query, top_k=20)
    context = "\n".join(docs)
    
    prompt = f"""The following are relevant documents:\n{context}\n\nQuery: {query}\nAnswer within 500 words.\nAnswer:"""
    
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    english_answer = response['message']['content']
    mongolian_answer = translate_rag_response(english_answer)
    return mongolian_answer
```

- Retrieves top 20 relevant documents per query
- Generates answers using Llama 3.2 via Ollama
- Translates responses to Mongolian



## Example and Usage

```python
query = 'Mongolian: Language courses how to apply and how expensive they are in Ireland'
answer = rag_pipeline(query)
print(answer)
```
> Ирланд дахь монголчуудын хэлний сургалтын талаарх мэдээллийг олохын тулд би янз бүрийн эх сурвалжуудаас хайсан.. Энд юу олсон бэ: ** Ирландын хэлний сургалт** Дублинд олон улсын оюутнуудад зориулсан, тэр дундаа Монголын оюутнуудад зориулсан англи хэлний сургалт явуулдаг хэд хэдэн сургууль байдаг. Хамгийн түгээмэл сонголт нь: 1. **IBAT College Dublin**: Насанд хүрэгчид болон хүүхдүүдэд зориулсан англи хэлний сургалтын өргөн хүрээг санал болгож байна.. 2 .. **Гриффит коллеж**: Англи хэлний сургалт, бизнесийн англи хэлний сургалт, шалгалтын бэлтгэл сургалт зэрэг сургалт явуулдаг. 3 .. ** Дублины хотын их сургуулийн хэлний төв**: Англи хэлний сургалт, сертификатын хөтөлбөр, англи хэлний сургалтын зэрэг олгодог. Ирланд улсад хэлний сургалтанд хамрагдахын тулд дараахь алхамуудыг дагах хэрэгтэй: 1. ** Сургуулийн цахим хуудсыг шалгаарай**: Таны сонирхож буй хэлний сургууль, коллежийн цахим хуудсыг очиж, одоогийн сургалт, үнэ, хүсэлт гаргах шаардлагыг мэдээрэй.. 2 .. ** Шаардлагатай баримт бичгийг цуглуулаарай**: Та хэн болохыг баталгаажуулах, боловсрол эзэмших, англи хэлний мэдлэгтэй байх шаардлагатай байж болно (хэрэв хэрэгжиж байгаа бол). 3 .. ** Онлайнаар эсвэл бие даан хүсэлт гаргах**: Онлайнаар хүсэлтээ ирүүлнэ үү эсвэл сургуулийн оффист очиж үйл явцыг дуусгах уу. **Хэрэг** Ирландын хэлний сургалтын үнэ нь сургууль, сургалтын төрөл, хугацаанаас хамаарч өөр өөр байдаг.. * **ИБАТ Колледж Дублин**: + Ерөнхий англи хэлний курс (20 цаг / долоо хоног): цагийн 15-20 евро + Англи хэлний нарийвчилсан курс (30 цаг / долоо хоног): цагийн 25-35 евро * **Гриффит Колледж**: + Англи хэлний курс (10-20 цаг / долоо хоног): цагийн 18-25 евро + Бизнесийн англи хэлний курс: цагийн 25-35 евро * **Дублин хотын их сургуулийн хэлний төв**: + Англи хэлний курс (10-20 цаг / долоо хоног): цагийн 15-20 евро ** нэмэлт өртөг** Сургалтын төлбөрөөс гадна та дараах бусад зардлыг авч үзэх шаардлагатай байж болно: * ** Амьдралын зардал**:. * **Виз олгох төлбөр**: Шаардлагатай бол виз олгох төлбөрийг төлөх хэрэгтэй (одоогийн байдлаар 50-100 евро). **Ирландын зарим хэлний сургуулиуд гадаадын оюутнуудад тэтгэлэг, санхүүгийн тусламж санал болгодог. Сургуулийн амжилт, хэлний ур чадвар, бусад шалгуур дээр тулгуурласан байж болно. Жишээ нь: * **IBAT College Dublin**: Олон улсын оюутнуудад олон төрлийн тэтгэлэг олгодог. * **Гриффит Колледж**: Санхүүгийн хэрэгцээг илтгэсэн олон улсын оюутнуудад тэтгэлэг олгодог. ** Эцсийн дүгнэлт** Ирланд дахь өөрийн хэрэгцээ, төсөвт нийцсэн хэлний сургалтыг олохын тулд дээрх сонголтуудыг судалж, сургуулиудад шууд хандахыг зөвлөж байна. Нэмэлт зардал, тэтгэлэг, санхүүгийн туслалцааны талаар мэдэхийг хичээгээрэй. Нэр дэвшүүлэхэд амжилт хүсье!.

Example output includes information about:
- Available language courses at IBAT, Griffith, and DCU
- Application procedures
- Cost ranges (15-35 EUR per hour)
- Scholarship opportunities
- Additional expenses (living costs, visa fees)

All responses are provided in Mongolian.

---

## Project Structure

```
visaTai/
├── data/
│   ├── data.txt           # Raw scraped data (combined)
│   ├── curated.txt        # Clustered knowledge base
│   └── *.parquet          # Living expenses (tabular)
├── pdfs/                  # Downloaded PDF documents
├── scraper.py             # Web scraping and OCR
├── process_data.py        # Clustering and curation
├── chatbot.py             # RAG pipeline
├── requirements.txt       # Dependencies
└── README.md
```

---

## Technical Stack

**Data Collection:**
- BeautifulSoup4 - HTML parsing
- Requests - HTTP client
- PyPDF2 - PDF processing
- Tesseract - OCR engine
- Pyarrow - Parquet format

**NLP and ML:**
- NLTK - Tokenization
- SentenceTransformers - Text embeddings
- Scikit-learn - DBSCAN clustering, cosine similarity
- NumPy - Numerical operations

**Generation:**
- Ollama - LLM inference
- Llama 3.2 - Language model

---

## Data Sources

- Irish Immigration Service (IIS) - Official visa information
- Numbeo - Living expenses data
- VFS Global Mongolia - Application procedures
- IBAT College Dublin - Language course information
- Griffith College - Educational programs
- Trinity College Dublin (TCD) - University information
- University College Cork (UCC) - University information

---

## Disclaimer

This tool provides information based on publicly available sources. Always verify immigration information with official Irish government sources and consult legal professionals for specific cases.
