import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Download, FileText } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { ReportFeatureList } from "../../components/ReportFeatureList";
import { ExplanationList } from "../../components/ExplanationList";
import { api } from "../../api/client";
import { downloadReportPdf, ReportPdfData } from "../../utils/reportPdf";
import { useDiagnosisFlow } from "./DiagnosisFlowContext";

export default function Results() {
  const navigate = useNavigate();
  const { result, reset } = useDiagnosisFlow();
  const [report, setReport] = useState<ReportPdfData | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  useEffect(() => {
    if (!result) return;
    api
      .getSessionReport(result.id)
      .then(setReport)
      .catch((err) => setReportError((err as Error).message));
  }, [result]);

  if (!result) {
    return (
      <AppShell title="Screening Result">
        <div className="text-center">
          <p className="text-ink-500">No result to show.</p>
          <button
            className="text-brand-600 underline mt-2 text-sm"
            onClick={() => navigate("/diagnose")}
          >
            Start a new diagnosis
          </button>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell title="Screening Result">
      <div className="max-w-6xl mx-auto grid lg:grid-cols-[minmax(0,320px)_1fr] gap-6 items-start">
        <Card className="p-8 text-center">
          <Badge tone={result.risk_category}>{result.risk_category} likelihood</Badge>
          <p className="mt-4 text-4xl font-bold text-ink-900">
            {(result.risk_score * 100).toFixed(0)}%
          </p>
          <p className="text-sm text-ink-400 mt-1">
            {result.session_type} · model {result.model_version}
          </p>

          <div className="flex flex-col gap-2 mt-8">
            <Button disabled={!report} onClick={() => report && downloadReportPdf(report)}>
              <Download size={16} />
              Download PDF Report
            </Button>
            <Button variant="secondary" onClick={() => navigate(`/reports/${result.id}`)}>
              <FileText size={16} />
              View Full Report
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                reset();
                navigate("/diagnose");
              }}
            >
              Run another screening
            </Button>
          </div>
        </Card>

        <div className="space-y-6 min-w-0">
          {report && report.summary_stats.length > 0 && (
            <Card className="p-6">
              <p className="font-medium text-ink-900 mb-4">Model insights</p>
              <dl className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {report.summary_stats.map((stat) => (
                  <div key={stat.label} className="bg-ink-50 rounded-lg px-3 py-2">
                    <dt className="text-xs text-ink-400">{stat.label}</dt>
                    <dd className="text-sm font-semibold text-ink-900">{stat.value}</dd>
                  </div>
                ))}
              </dl>
            </Card>
          )}

          {report && (report.explanation.length > 0 || report.explanation_note) && (
            <Card className="p-6">
              <p className="font-medium text-ink-900 mb-4">What influenced this result</p>
              <ExplanationList items={report.explanation} note={report.explanation_note} />
            </Card>
          )}

          <Card className="p-6">
            <p className="font-medium text-ink-900 mb-4">Screening Report</p>
            {reportError && <p className="text-red-600 text-sm">{reportError}</p>}
            {!report && !reportError && <p className="text-ink-400 text-sm">Generating report…</p>}
            {report && <ReportFeatureList features={report.features} note={report.note} />}
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
