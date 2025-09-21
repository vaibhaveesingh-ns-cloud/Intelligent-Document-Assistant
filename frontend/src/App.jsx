import { useCallback, useEffect, useMemo, useState } from 'react';
import Header from './components/Header.jsx';
import UploadSection from './components/UploadSection.jsx';
import ConversationPanel from './components/ConversationPanel.jsx';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const createId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
};

const parseEmbeddingSelection = (value) => {
  const [provider, ...rest] = value.split(':');
  return { provider, model: rest.join(':') };
};

function App() {
  const [sessionId] = useState(() => createId());
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [embeddingOption, setEmbeddingOption] = useState('openai:text-embedding-3-small');
  const [llmModel, setLlmModel] = useState('gpt-3.5-turbo');
  const [apiKey, setApiKey] = useState('');

  const acceptTypes = useMemo(
    () => '.pdf,.docx,.txt,.md,.pptx,.csv,.png,.jpg,.jpeg',
    []
  );

  useEffect(() => {
    const loadDefaults = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/config`);
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        if (data?.defaults) {
          setChunkSize(data.defaults.chunk_size ?? 1000);
          setChunkOverlap(data.defaults.chunk_overlap ?? 200);
          if (data.defaults.embedding_model) {
            setEmbeddingOption(`openai:${data.defaults.embedding_model}`);
          }
          if (data.defaults.llm_model) {
            setLlmModel(data.defaults.llm_model);
          }
        }
      } catch (error) {
        // ignore; backend might not be running during static preview
      }
    };
    loadDefaults();
  }, []);

  const onFilesAdded = useCallback((newFiles) => {
    setFiles((previous) => {
      const existingNames = new Set(previous.map((file) => file.name));
      const filtered = newFiles.filter((file) => !existingNames.has(file.name));
      if (!filtered.length) {
        return previous;
      }
      return [...previous, ...filtered];
    });
  }, []);

  const onRemoveFile = useCallback((name) => {
    setFiles((previous) => previous.filter((file) => file.name !== name));
  }, []);

  const onClearFiles = useCallback(() => {
    setFiles([]);
  }, []);

  const handleUpload = useCallback(async () => {
    if (!files.length) {
      setStatus({ type: 'error', message: 'Add at least one document before uploading.' });
      return;
    }

    setIsUploading(true);
    setStatus(null);

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    formData.append('chunk_size', String(chunkSize));
    formData.append('chunk_overlap', String(chunkOverlap));

    const { provider, model } = parseEmbeddingSelection(embeddingOption);
    if (provider === 'local') {
      formData.append('use_local_embeddings', 'true');
      formData.append('local_embedding_model', model);
    } else {
      formData.append('use_local_embeddings', 'false');
      formData.append('embedding_model', model);
      if (apiKey) {
        formData.append('openai_api_key', apiKey);
      }
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'Unable to upload documents.');
      }
      const result = await response.json();
      setStatus({
        type: 'success',
        message: `Indexed ${result.ingested} chunks${result.skipped ? `, skipped ${result.skipped}` : ''}.`
      });
    } catch (error) {
      setStatus({ type: 'error', message: error.message || 'Upload failed. Check the API server.' });
    } finally {
      setIsUploading(false);
    }
  }, [apiKey, chunkOverlap, chunkSize, embeddingOption, files]);

  const handleSendQuestion = useCallback(
    async (question) => {
      setStatus(null);
      const userMessage = { id: createId(), role: 'user', content: question };
      setMessages((previous) => [...previous, userMessage]);
      setIsSending(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            question,
            session_id: sessionId,
            top_k: 4,
            llm_model: llmModel,
            openai_api_key: apiKey || undefined
          })
        });
        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw new Error(error.detail || 'The assistant was unable to answer.');
        }
        const payload = await response.json();
        const assistantMessage = {
          id: createId(),
          role: 'assistant',
          content: payload.answer,
          sources: payload.sources
        };
        setMessages((previous) => [...previous, assistantMessage]);
      } catch (error) {
        setStatus({ type: 'error', message: error.message || 'Unable to contact the assistant.' });
      } finally {
        setIsSending(false);
      }
    },
    [apiKey, llmModel, sessionId]
  );

  return (
    <div className="app-container">
      <Header />
      <main>
        <UploadSection
          files={files}
          onFilesAdded={onFilesAdded}
          onRemoveFile={onRemoveFile}
          onClearFiles={onClearFiles}
          onUpload={handleUpload}
          isUploading={isUploading}
          chunkSize={chunkSize}
          setChunkSize={setChunkSize}
          chunkOverlap={chunkOverlap}
          setChunkOverlap={setChunkOverlap}
          embeddingOption={embeddingOption}
          setEmbeddingOption={setEmbeddingOption}
          llmModel={llmModel}
          setLlmModel={setLlmModel}
          apiKey={apiKey}
          setApiKey={setApiKey}
          status={status}
          accept={acceptTypes}
        />
        <ConversationPanel
          messages={messages}
          onSendQuestion={handleSendQuestion}
          isSending={isSending}
        />
      </main>
    </div>
  );
}

export default App;
