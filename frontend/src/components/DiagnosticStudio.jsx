// Zone order must match the backend's zone_breakdown() nesting exactly:
// for row in ZONE_ROWS: for col in ZONE_COLS. With a 2-col x 3-row CSS grid,
// row-major auto-placement lines each returned zone up with the correct cell
// automatically — no manual coordinate math needed on the frontend.
function ZoneGridOverlay({ zones }) {
  if (!zones || zones.length === 0) return null;
  const maxScore = Math.max(...zones.map((z) => z.activation), 1);

  return (
    <div className="zone-grid-overlay">
      {zones.map((z) => (
        <div
          key={z.zone}
          className={`zone-cell${z.activation === maxScore ? " hot" : ""}`}
        >
          <span className="zone-label">
            {z.zone} · <span className="zone-score">{z.activation.toFixed(0)}%</span>
          </span>
        </div>
      ))}
    </div>
  );
}

function ZoneStrip({ zones }) {
  if (!zones || zones.length === 0) return null;
  return (
    <div className="zone-strip">
      {zones.map((z) => (
        <div className="zone-chip" key={z.zone}>
          <div className="label">{z.zone.toUpperCase()}</div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${Math.min(z.activation, 100)}%` }} />
          </div>
          <div className="value">{z.activation.toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
}

export default function DiagnosticStudio({ previewUrl, result, analyzing, error }) {
  const displayImage = result?.heatmapOverlay || previewUrl;

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <span className="dot" />
          Diagnostic Studio
        </div>
        <span className="panel-sub">
          {result ? `${result.numClasses}-class model · zone-wise Grad-CAM` : "awaiting scan"}
        </span>
      </div>

      <div className="viewer-body">
        <div className="viewer-frame">
          {displayImage ? (
            <img src={displayImage} alt="Chest X-ray with Grad-CAM overlay" />
          ) : (
            <div className="viewer-empty">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4d5975" strokeWidth="1.3">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M8 12h8M12 8v8" strokeLinecap="round" />
              </svg>
              <span>Load a chest X-ray to begin analysis</span>
            </div>
          )}

          {result && <ZoneGridOverlay zones={result.primaryZones} />}

          {analyzing && (
            <div className="viewer-loading">
              <div className="scan-line" />
              <span>Running DenseNet121 inference &amp; Grad-CAM…</span>
            </div>
          )}
        </div>
      </div>

      {error && <div className="viewer-error">{error}</div>}

      {result && <ZoneStrip zones={result.primaryZones} />}
    </div>
  );
}
