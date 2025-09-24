import { useCallback, useState } from 'react';
import StatusBanner from './StatusBanner.jsx';
import '../App.css';

function UploadSection({
  files,
  onFilesAdded,
  onRemoveFile,
  onClearFiles,
  onUpload,
  isUploading,
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
