const SEVERITY_COLOR = {
  Critical: "var(--critical)",
  Severe: "var(--severe)",
  Moderate: "var(--moderate)",
  Mild: "var(--mild)",
  Normal: "var(--normal)",
};

export default function FindingsPanel({ result, pathologiesScreened }) {
  const findings = result?.findings || [];
  const probabilities = result?.allProbabilities || {};
  const thresholds = result?.classThresholds || {};

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
            {result
              ? "No class cleared its decision threshold for this scan. See all probabilities below."
              : "No scan analyzed yet. Findings will appear here, ranked by model confidence."}
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

      {Object.keys(probabilities).length > 0 && (
        <details className="probability-details">
          <summary>All class probabilities and decision thresholds</summary>
          <div className="probability-list">
            {Object.entries(probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([name, probability]) => {
                const threshold = thresholds[name] ?? 0.5;
                const detected = probability >= threshold;
                return (
                  <div className="probability-row" key={name}>
                    <span>{name}</span>
                    <span>{Math.round(probability * 100)}%</span>
                    <span className={detected ? "probability-detected" : "probability-negative"}>
                      {detected ? "Above threshold" : "Below threshold"}
                    </span>
                    <span>Cutoff {Math.round(threshold * 100)}%</span>
                  </div>
                );
              })}
          </div>
        </details>
      )}

      {pathologiesScreened && (
        <div className="screened-list">
          Screened for: {pathologiesScreened.join(" · ")}
        </div>
      )}
    </div>
  );
}
