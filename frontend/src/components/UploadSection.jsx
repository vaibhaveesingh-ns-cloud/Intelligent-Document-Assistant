import { useCallback, useState } from 'react';
import StatusBanner from './StatusBanner.jsx';
import '../App.css';

const EMBEDDING_OPTIONS = [
  { value: 'openai:text-embedding-3-small', label: 'OpenAI · text-embedding-3-small' },
  { value: 'openai:text-embedding-3-large', label: 'OpenAI · text-embedding-3-large' },
  { value: 'local:sentence-transformers/all-MiniLM-L6-v2', label: 'Local · all-MiniLM-L6-v2' }
];

function UploadSection({
  files,
  onFilesAdded,
  onRemoveFile,
  onClearFiles,
  onUpload,
  isUploading,
  chunkSize,
  setChunkSize,
  chunkOverlap,
  setChunkOverlap,
  embeddingOption,
  setEmbeddingOption,
  llmModel,
  setLlmModel,
  apiKey,
  setApiKey,
  status,
  accept
}) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragOver(false);
      const newFiles = Array.from(event.dataTransfer.files || []);
      if (newFiles.length) {
        onFilesAdded(newFiles);
      }
    },
    [onFilesAdded]
  );

  const handleBrowse = useCallback((event) => {
    const newFiles = Array.from(event.target.files || []);
    if (newFiles.length) {
      onFilesAdded(newFiles);
    }
  }, [onFilesAdded]);

  return (
    <section className="section" id="get-started">
      <div className="hero">
        <h1>Upload Your Documents</h1>
        <p>Drag and drop files here, or click to browse. We support PDF, DOCX, and TXT formats.</p>

        <div
          className={`upload-card ${isDragOver ? 'dragover' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="upload-icon">↑</div>
          <h3>Drag & drop files here</h3>
          <p>or</p>
          <div className="upload-buttons">
            <label className="primary-button" htmlFor="file-input">
              Browse Files
            </label>
            <input
              id="file-input"
              type="file"
              multiple
              accept={accept}
              onChange={handleBrowse}
              style={{ display: 'none' }}
            />
            <button
              type="button"
              className="secondary-button"
              onClick={onClearFiles}
              disabled={!files.length}
            >
              Clear
            </button>
          </div>
        </div>

        {files.length > 0 && (
          <div className="file-list">
            {files.map((file) => (
              <div key={file.name} className="file-pill">
                <span>{file.name}</span>
                <button type="button" onClick={() => onRemoveFile(file.name)}>
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="settings-grid">
          <div className="settings-field">
            <label htmlFor="embedding">Embeddings</label>
            <select
              id="embedding"
              value={embeddingOption}
              onChange={(event) => setEmbeddingOption(event.target.value)}
            >
              {EMBEDDING_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <p className="settings-footnote">
              Local embeddings avoid API usage. OpenAI options require an API key.
            </p>
          </div>

          <div className="settings-field">
            <label htmlFor="chunk-size">Chunk size</label>
            <input
              id="chunk-size"
              type="number"
              min="200"
              max="2000"
              value={chunkSize}
              onChange={(event) => setChunkSize(Number(event.target.value))}
            />
          </div>

          <div className="settings-field">
            <label htmlFor="chunk-overlap">Chunk overlap</label>
            <input
              id="chunk-overlap"
              type="number"
              min="0"
              max="500"
              value={chunkOverlap}
              onChange={(event) => setChunkOverlap(Number(event.target.value))}
            />
          </div>

          <div className="settings-field">
            <label htmlFor="llm-model">Chat model</label>
            <input
              id="llm-model"
              type="text"
              value={llmModel}
              onChange={(event) => setLlmModel(event.target.value)}
            />
            <p className="settings-footnote">Responses are generated using the configured ChatGPT model.</p>
          </div>

          <div className="settings-field" style={{ gridColumn: '1 / -1' }}>
            <label htmlFor="api-key">OpenAI API key</label>
            <input
              id="api-key"
              type="password"
              placeholder="sk-..."
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
            />
            <p className="settings-footnote">
              Your key is only used to call OpenAI during ingestion and chat. It is never stored on the server.
            </p>
          </div>
        </div>

        <div className="upload-buttons" style={{ justifyContent: 'flex-start', marginTop: '2rem' }}>
          <button type="button" className="primary-button" onClick={onUpload} disabled={isUploading}>
            {isUploading ? 'Uploading…' : 'Upload and Index'}
          </button>
        </div>

        <StatusBanner status={status} />
      </div>
    </section>
  );
}

export default UploadSection;
