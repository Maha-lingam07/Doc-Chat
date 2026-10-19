const byId = (id) => document.getElementById(id);
const state = { indexName: '', busy: false };
const fileInput = byId('fileInput');
const indexInput = byId('indexName');
const questionInput = byId('question');
const messages = byId('messages');

function notify(message) {
  const toast = byId('toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => toast.classList.remove('show'), 4200);
}

function setStatus(message) { byId('indexStatus').textContent = message; }
function setActiveIndex(name) {
  state.indexName = name;
  questionInput.disabled = false;
  questionInput.placeholder = `Ask a question about ${name}…`;
  byId('send').disabled = !questionInput.value.trim() || state.busy;
  byId('welcome').hidden = true;
  setStatus(`Connected to index “${name}”.`);
}

async function responseJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || 'The request could not be completed.');
  return body;
}

async function openIndex() {
  const name = indexInput.value.trim();
  if (!name) return notify('Enter a Pinecone index name first.');
  state.busy = true;
  byId('openIndex').disabled = true;
  setStatus('Checking Pinecone…');
  try {
    const result = await responseJson(await fetch('/api/index/select', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ index_name: name }),
    }));
    setActiveIndex(result.index_name);
    setStatus(`Connected to “${name}” · ${result.vector_count.toLocaleString()} vectors`);
    messages.replaceChildren();
  } catch (error) {
    setStatus('');
    notify(error.message);
  } finally {
    state.busy = false;
    byId('openIndex').disabled = false;
    byId('send').disabled = !state.indexName || !questionInput.value.trim();
  }
}

byId('openIndex').addEventListener('click', openIndex);
indexInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); openIndex(); } });

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  fileInput.value = '';
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) return notify('Choose a PDF file.');
  if (file.size > 30 * 1024 * 1024) return notify('PDFs must be 30 MB or smaller.');
  const name = indexInput.value.trim();
  if (!name) return notify('Enter a Pinecone index name before uploading.');

  state.busy = true;
  setStatus('Checking the index and processing the PDF if needed…');
  try {
    const body = new FormData();
    body.append('file', file);
    body.append('index_name', name);
    const result = await responseJson(await fetch('/api/documents', { method: 'POST', body }));
    setActiveIndex(result.index_name);
    messages.replaceChildren();
    if (result.existing) {
      setStatus(`Loaded “${name}” · ${result.vector_count.toLocaleString()} vectors. PDF processing skipped.`);
    } else {
      setStatus(`Created “${name}” and indexed ${result.chunks} text chunks from ${result.pages} pages.`);
    }
  } catch (error) {
    setStatus('');
    notify(error.message);
  } finally {
    state.busy = false;
    byId('send').disabled = !state.indexName || !questionInput.value.trim();
  }
});

function addMessage(text, role, sources = []) {
  const item = document.createElement('article');
  item.className = `message ${role}`;
  item.textContent = text;
  if (sources.length) {
    const sourceList = document.createElement('div');
    sourceList.className = 'sources';
    for (const source of sources) {
      const chip = document.createElement('span');
      chip.className = 'source';
      chip.textContent = source.page === '?' ? 'Saved passage' : `Page ${source.page}`;
      sourceList.append(chip);
    }
    item.append(sourceList);
  }
  messages.append(item);
  byId('conversation').scrollTo({ top: byId('conversation').scrollHeight, behavior: 'smooth' });
  return item;
}

async function sendQuestion(question) {
  if (!question || !state.indexName || state.busy) return;
  state.busy = true;
  questionInput.value = '';
  questionInput.style.height = '40px';
  byId('send').disabled = true;
  addMessage(question, 'user');
  const pending = addMessage('Searching the index…', 'assistant');
  try {
    const result = await responseJson(await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index_name: state.indexName, question }),
    }));
    pending.remove();
    addMessage(result.answer, 'assistant', result.sources || []);
  } catch (error) {
    pending.textContent = error.message;
  } finally {
    state.busy = false;
    byId('send').disabled = !state.indexName || !questionInput.value.trim();
  }
}

byId('chatForm').addEventListener('submit', (event) => {
  event.preventDefault();
  sendQuestion(questionInput.value.trim());
});
questionInput.addEventListener('input', () => {
  questionInput.style.height = '40px';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
  byId('send').disabled = !state.indexName || state.busy || !questionInput.value.trim();
});
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); byId('chatForm').requestSubmit(); }
});

fetch('/api/config').then(responseJson).then((config) => { indexInput.value = config.default_index || ''; }).catch(() => {});
