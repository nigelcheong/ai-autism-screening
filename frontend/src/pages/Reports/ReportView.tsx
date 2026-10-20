import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Printer, Download, FileCheck2 } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ReportFeatureList } from "../../components/ReportFeatureList";
import { ExplanationList } from "../../components/ExplanationList";
import { api } from "../../api/client";
import { downloadReportPdf, ReportPdfData } from "../../utils/reportPdf";

type Report = ReportPdfData;

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-ink-400">{label}</p>
      <p className="text-sm font-medium text-ink-900">{value}</p>
    </div>
  );
}

export default function ReportView() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    api
      .getSessionReport(sessionId)
      .then(setReport)
      .catch((err) => setError((err as Error).message));
  }, [sessionId]);

  return (
    <AppShell title="Screening Report">
      <div className="max-w-5xl mx-auto">
        {error && <p className="text-red-600 text-sm mb-4">{error}</p>}
        {!report && !error && <p className="text-ink-400 text-sm">Loading report…</p>}

        {report && (
          <>
            <div className="flex justify-end gap-2 mb-4 no-print">
              <Button size="sm" onClick={() => downloadReportPdf(report)}>
                <Download size={14} />
                Download PDF
              </Button>
              <Button variant="secondary" size="sm" onClick={() => window.print()}>
                <Printer size={14} />
                Print
              </Button>
            </div>

            <Card className="p-6 mb-6">
              <div className="flex items-center justify-between flex-wrap gap-2 pb-4 mb-4 border-b border-ink-100">
                <div>
                  <p className="text-xs font-semibold tracking-wide text-brand-600 uppercase">
                    {report.hospital_name}
                  </p>
                  <p className="font-semibold text-ink-900 text-lg">Autism Screening Report</p>
                </div>
                <p className="text-xs text-ink-400">
                  Screened {new Date(report.created_at).toLocaleDateString()}
                </p>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                <Field label="Patient" value={report.patient_name} />
                <Field
                  label="Date of birth"
                  value={
                    report.patient_dob
                      ? `${new Date(report.patient_dob).toLocaleDateString()}${
                          report.patient_age_label ? ` (${report.patient_age_label} old)` : ""
                        }`
                      : "—"
                  }
                />
                <Field label="Patient ID" value={report.patient_external_id ?? "—"} />
                <Field label="Reviewing clinician" value={report.doctor_email ?? "Unassigned"} />
                <Field
                  label="Screening date"
                  value={new Date(report.created_at).toLocaleString()}
                />
                <Field
                  label="Screening type"
                  value={report.session_type === "image" ? "Facial image analysis" : "Questionnaire"}
                />
              </div>

              <div className="flex items-center justify-between flex-wrap gap-2 bg-ink-50 rounded-lg px-4 py-3">
                <p className="text-sm font-medium text-ink-700">Screening result</p>
                {report.risk_category && (
                  <div className="text-right">
                    <Badge tone={report.risk_category as any}>{report.risk_category} likelihood</Badge>
                    {report.risk_score !== null && (
                      <p className="text-2xl font-bold text-ink-900 mt-1">
                        {(report.risk_score * 100).toFixed(0)}%
                      </p>
                    )}
                  </div>
                )}
              </div>
            </Card>

            {report.summary_stats.length > 0 && (
              <Card className="p-6 mb-6">
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

            {(report.explanation.length > 0 || report.explanation_note) && (
              <Card className="p-6 mb-6">
                <p className="font-medium text-ink-900 mb-4">What influenced this result</p>
                <ExplanationList items={report.explanation} note={report.explanation_note} />
              </Card>
            )}

            <Card className="p-6 mb-6">
              <p className="font-medium text-ink-900 mb-4">Features considered</p>
              <ReportFeatureList features={report.features} note={report.note} />
            </Card>

            <Card className="p-5 text-sm text-ink-500">
              <div className="flex items-start gap-3">
                <FileCheck2 size={18} className="text-ink-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-ink-700 font-medium">Scored by: {report.model_description}</p>
                  <p className="text-xs text-ink-400 mt-1">
                    Model reference: {report.model_version ?? "—"} · Report generated{" "}
                    {new Date(report.generated_at).toLocaleString()}
                  </p>
                  <p className="text-xs text-ink-400 mt-2">
                    This is a screening aid, not a clinical diagnosis. Please consult a qualified
                    healthcare professional for a full evaluation.
                  </p>
                </div>
              </div>
            </Card>
          </>
        )}
      </div>
    </AppShell>
  );
}
