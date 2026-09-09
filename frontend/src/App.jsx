import { useEffect, useState } from "react";
import UploadModal from "./components/UploadModal.jsx";
import DiagnosticStudio from "./components/DiagnosticStudio.jsx";
import FindingsPanel from "./components/FindingsPanel.jsx";
import ClinicalReport from "./components/ClinicalReport.jsx";

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("studio");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ ready: false, error: "Cannot reach backend at :5000" }));
  }, []);

  async function handleAnalyze(file, localPreviewUrl) {
    setShowModal(false);
    setPreviewUrl(localPreviewUrl);
    setResult(null);
    setError(null);
    setAnalyzing(true);

    try {
      const imageBase64 = await fileToBase64(file);
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64 }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.details || data.error || "Analysis failed.");
      } else {
        setResult(data);
      }
    } catch (e) {
      setError("Could not reach the analysis backend. Is the Flask server running on :5000?");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">X</div>
          <div className="brand-text">
            <h1>ExplainX-Ray</h1>
            <p>DenseNet121 · Real Grad-CAM · Zone-wise Analysis</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="status-pill">
            <span
              className={`status-dot${
                health?.ready ? " ready" : health?.error ? " error" : ""
              }`}
            />
            {health === null
              ? "Checking model…"
              : health.ready
              ? `Model ready · ${health.numClasses} classes · ${health.device}`
              : "Model not trained"}
          </div>
          <button className="btn-primary" onClick={() => setShowModal(true)}>
            Load X-ray scan
          </button>
        </div>
      </header>

      <nav className="tab-bar" aria-label="Analysis view">
        <button className={`btn-secondary${activeTab === "studio" ? " active" : ""}`} onClick={() => setActiveTab("studio")}>Diagnostic studio</button>
        <button className={`btn-secondary${activeTab === "report" ? " active" : ""}`} onClick={() => setActiveTab("report")}>Clinical report</button>
      </nav>

      {activeTab === "studio" ? (
        <main className="studio">
          <DiagnosticStudio previewUrl={previewUrl} result={result} analyzing={analyzing} error={error} />
          <FindingsPanel result={result} pathologiesScreened={health?.pathologies} />
        </main>
      ) : (
        <main className="report-page"><ClinicalReport result={result} /></main>
      )}

      {showModal && (
        <UploadModal onClose={() => setShowModal(false)} onAnalyze={handleAnalyze} />
      )}
    </div>
  );
}
