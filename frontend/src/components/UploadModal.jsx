import { useRef, useState } from "react";

export default function UploadModal({ onClose, onAnalyze }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFile(f) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal-card" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Load chest X-ray</h2>
            <p>Upload a PA/AP chest radiograph — PNG, JPG, or WEBP.</p>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#2dd4e8" strokeWidth="1.6">
            <path d="M12 16V4M12 4L7 9M12 4l5 5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p className="main">Drag &amp; drop chest X-ray here</p>
          <p className="sub">or click to browse your files</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        {file && (
          <div className="modal-preview">
            <img src={previewUrl} alt="Selected X-ray preview" />
            <div>
              <div className="file-name">{file.name}</div>
              <div className="file-meta">{(file.size / 1024).toFixed(0)} KB</div>
            </div>
          </div>
        )}

        <div className="modal-actions">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-analyze"
            disabled={!file}
            onClick={() => file && onAnalyze(file, previewUrl)}
          >
            Analyze X-ray
          </button>
        </div>
      </div>
    </div>
  );
}
