const SEVERITY_COLOR = {
  Critical: "var(--critical)",
  Severe: "var(--severe)",
  Moderate: "var(--moderate)",
  Mild: "var(--mild)",
  Normal: "var(--normal)",
};

export default function FindingsPanel({ result, pathologiesScreened }) {
  const findings = result?.findings || [];

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <span className="dot" />
          Detected Pathologies {findings.length > 0 && `(${findings.length})`}
        </div>
      </div>

      <div className="findings-body">
        {findings.length === 0 ? (
          <div className="findings-empty">
            No scan analyzed yet. Findings will appear here, ranked by model confidence.
          </div>
        ) : (
          findings.map((f) => (
            <div className="finding-card" key={f.id}>
              <div className="finding-top">
                <div className="finding-name">{f.name}</div>
                <div className="finding-prob">{Math.round(f.probability * 100)}%</div>
              </div>

              <div className="finding-bar-track">
                <div
                  className="finding-bar-fill"
                  style={{
                    width: `${Math.round(f.probability * 100)}%`,
                    background: SEVERITY_COLOR[f.severity] || "var(--accent)",
                  }}
                />
              </div>

              <div className="finding-meta">
                <span>
                  <span className={`severity-tag ${f.severity}`}>{f.severity}</span>
                </span>
                <span>{f.anatomicalRegion}</span>
              </div>
              <div className="finding-meta" style={{ marginTop: 6 }}>
                <span>Peak zone</span>
                <span>{f.primaryZone}</span>
              </div>
              <div className="finding-meta" style={{ marginTop: 6 }}>
                <span>ASFI score</span>
                <span>{f.asfi?.score ?? "—"}/100</span>
              </div>
              <div className="finding-meta" style={{ marginTop: 6 }}>
                <span>Pattern</span>
                <span>{f.distribution?.pattern ?? "—"}</span>
              </div>
            </div>
          ))
        )}
      </div>

      {pathologiesScreened && (
        <div className="screened-list">
          Screened for: {pathologiesScreened.join(" · ")}
        </div>
      )}
    </div>
  );
}
