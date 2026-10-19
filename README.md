# Doc-Chat

**Ask questions about your PDFs and get clear, source-grounded answers.** Doc-Chat is a local web app that extracts PDF text, creates semantic embeddings, stores them in Pinecone, and uses Google Gemini to answer questions with page citations.

## What you can do

- Upload a selectable-text PDF and index its content for semantic search.
- Ask questions in a simple chat interface and receive answers grounded in retrieved passages.
- Cite supporting PDF page numbers when available.
- Reopen an existing Pinecone index by name. Existing indexes are loaded without reprocessing an uploaded PDF.
- Generate embeddings with PyTorch on a compatible Windows DirectML GPU, including supported AMD GPUs, with CPU fallback when DirectML is unavailable.

## How it works

```text
PDF → extract page text → split into chunks → create embeddings → Pinecone
                                                                    ↓
Question → create query embedding → retrieve relevant passages → Gemini answer
```

The browser UI is served by FastAPI. Pinecone stores each PDF's vectors and text metadata in a document-specific namespace. The app retrieves the most relevant passages and sends them to Gemini as context for the answer.

## Requirements

- Windows 10 or later for DirectML GPU acceleration; CPU mode is also supported.
- Python 3.10–3.12.
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey).
- A [Pinecone API key](https://app.pinecone.io/).
- Internet access for API requests and the first download of the embedding model.

> The first model download can take a few minutes. Hugging Face may warn about unauthenticated downloads; this is a rate-limit notice. You can optionally set `HF_TOKEN` in your environment to use an authenticated Hugging Face account.

## Setup and run (Windows PowerShell)

Open PowerShell in the project folder and run:

```powershell
cd "C:\Users\MAHALINGAM\Desktop\wd\Projects\DocChat"
py -3.12 -m venv .venv-docchat
.\.venv-docchat\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual environment activation, allow it for the current terminal session, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-docchat\Scripts\Activate.ps1
```

Open `.env` and replace the placeholder values with your API keys:

```dotenv
GEMINI_API_KEY=your_google_ai_studio_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Keep `.env` private; it contains credentials and is excluded from Git. Start the app:

```powershell
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> in your browser. To stop the server, press **Ctrl+C** in PowerShell. If port `8080` is unavailable, choose another port, such as `8081`, and use that port in the browser address.

## First-use tutorial

1. Start Doc-Chat and open it in your browser.
2. In **Pinecone index name**, enter a new name using lowercase letters, numbers, and hyphens (for example, `my-first-pdf`). Names must start with a letter or number and be no longer than 45 characters.
3. Select **Upload PDF** and choose a PDF with selectable text. Doc-Chat creates the Pinecone index, extracts and chunks the text, and stores the embeddings and page metadata.
4. When indexing finishes, type a question in the chat box and press **Enter** or the send button. The app searches the active index and asks Gemini to answer from the matching passages.
5. Later, enter that same index name and select **Open index** to chat with the saved vectors without re-uploading or re-embedding the PDF.

If the index name already exists when you upload, Doc-Chat opens that index and skips PDF parsing, chunking, and embedding. Make sure the index contains vectors created by Doc-Chat and uses the same embedding model and vector dimension.

## GPU and embedding settings

On Windows, `requirements.txt` installs the DirectML PyTorch package. With `EMBEDDING_DEVICE=auto`, Doc-Chat uses DirectML when available and falls back to CPU otherwise. On startup, check the terminal for a line such as `Loading embedding model all-mpnet-base-v2 on ...` to see the selected device.

Set one of these values in `.env` to choose the device explicitly:

```dotenv
EMBEDDING_DEVICE=auto
```

- `auto` — use DirectML when available; otherwise use CPU.
- `directml` — require DirectML; startup reports an error if it is unavailable.
- `cpu` — always run embeddings on the CPU.

DirectML performance and compatibility depend on your GPU, driver, Windows version, Python version, and installed PyTorch packages. If startup reports `DirectML unavailable ...; using CPU for embeddings`, check that the virtual environment is active and reinstall the project requirements with `python -m pip install -r requirements.txt`.

## Configuration

All settings are read from `.env`. The defaults are defined in `.env.example`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Google Gemini API key for answers | Required |
| `PINECONE_API_KEY` | Pinecone API key for vector storage and search | Required |
| `PINECONE_INDEX` | Index name prefilled in the app | `docchat-index` |
| `PINECONE_CLOUD` | Pinecone serverless cloud provider | `aws` |
| `PINECONE_REGION` | Pinecone serverless region | `us-east-1` |
| `GEMINI_MODEL` | Gemini model used to generate answers | `gemini-3.8-flash` |
| `EMBEDDING_MODEL` | Sentence Transformers model used for vectors | `all-mpnet-base-v2` |
| `EMBEDDING_DEVICE` | Embedding device: `auto`, `directml`, or `cpu` | `auto` |
| `DOCCHAT_DATA_DIR` | Folder for uploaded PDFs and local document metadata | `./data` |

The default embedding model produces 768-dimensional vectors. Any Pinecone index used with Doc-Chat must have a matching dimension and compatible embedding model.

## Data and PDF notes

- PDFs are limited to **30 MB**.
- PDFs must contain selectable text. Scanned image-only PDFs need OCR before upload.
- Local uploads are saved under `data/uploads/`; local document metadata is saved in `data/documents.json`.
- Vector data and extracted text metadata are stored in Pinecone. Each uploaded document is assigned a namespace.
- The `data/` directory is ignored by Git because it contains local documents and application data.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `Could not answer ... GEMINI_API_KEY` | Add a valid `GEMINI_API_KEY` to `.env`, then restart Doc-Chat. |
| `Could not answer ... PINECONE_API_KEY` | Add a valid `PINECONE_API_KEY` to `.env`, then restart Doc-Chat. |
| DirectML unavailable; using CPU | Activate `.venv-docchat`, confirm you are on Windows with a supported GPU/driver, and install `requirements.txt` in that environment. CPU fallback still allows the app to run. |
| Pinecone index not found | Check the spelling and confirm the index exists in the Pinecone project associated with your API key. |
| No selectable text was found | The PDF may be a scan; run OCR to create a searchable PDF and upload it again. |
| Port is already in use or access is denied | Stop the other server process or choose a different port in the Uvicorn command. |

## Project files

```text
DocChat/
├── app.py                 # FastAPI endpoints, PDF indexing, embeddings, and chat
├── static/
│   ├── index.html          # App page
│   ├── styles.css          # Responsive styles
│   ├── app.js              # Upload, index selection, and chat interactions
│   └── doc-chat-brand.jpg  # App logo
├── .env.example            # Configuration template (safe to commit)
├── requirements.txt        # Python dependencies
└── README.md               # Setup and usage guide
```
## ScreenShot of the UI

<img width="1919" height="908" alt="SS" src="https://github.com/user-attachments/assets/1ce15a04-b7ff-43de-a923-dfd98e942426" />
