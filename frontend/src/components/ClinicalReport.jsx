export default function ClinicalReport({ result }) {
  const report = result?.clinicalReport;

  if (!report) {
    return (
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span className="dot" />Clinical Report</div>
        </div>
        <div className="findings-empty">Run an analysis to generate a report.</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span className="dot" />Clinical Report</div>
      </div>
      <div className="report-body">
        <p className="report-summary">{report.summary}</p>
        {report.findings.map((finding) => (
          <div className="report-finding" key={finding.name}>
            <h4>{finding.name}</h4>
            <p>{finding.explanation}</p>
          </div>
        ))}
        <div className="report-methodology">
          <h4>Methodology</h4>
          <p>{report.methodology}</p>
        </div>
        <div className="report-disclaimer">{report.disclaimer}</div>
      </div>
    </div>
  );
}
