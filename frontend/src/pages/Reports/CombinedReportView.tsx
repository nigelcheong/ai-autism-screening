import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft, FileCheck2, Pencil } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ExplanationList } from "../../components/ExplanationList";
import { ReportFeatureList } from "../../components/ReportFeatureList";
import { api, CombinedReportData } from "../../api/client";

export default function CombinedReportView() {
  const { reportId } = useParams<{ reportId: string }>();
  const [report, setReport] = useState<CombinedReportData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editingComments, setEditingComments] = useState(false);
  const [commentsInput, setCommentsInput] = useState("");
  const [saving, setSaving] = useState(false);

  function load() {
    if (!reportId) return;
    api
      .getCombinedReport(reportId)
      .then(setReport)
      .catch((err) => setError((err as Error).message));
  }

  useEffect(load, [reportId]);

  async function saveComments() {
    if (!reportId) return;
    setSaving(true);
    try {
      await api.updateCombinedReportComments(reportId, commentsInput);
      setEditingComments(false);
      load();
    } finally {
      setSaving(false);
    }
  }

  if (error) {
    return (
      <AppShell title="Combined Report">
        <p className="text-red-600 text-sm">{error}</p>
      </AppShell>
    );
  }

  if (!report) {
    return (
      <AppShell title="Combined Report">
        <p className="text-ink-400 text-sm">Loading…</p>
      </AppShell>
    );
  }

  return (
    <AppShell title="Combined Report">
      <div className="max-w-5xl mx-auto">
        <Link
          to={`/patients`}
          className="inline-flex items-center gap-1 text-sm text-ink-500 hover:text-ink-900 mb-4"
        >
          <ChevronLeft size={16} />
          All patients
        </Link>

        <Card className="p-6 mb-6">
          <p className="text-xs text-ink-400">PATIENT</p>
          <p className="text-lg font-semibold text-ink-900">{report.patient_name}</p>
          <p className="text-sm text-ink-500 mt-1">
            {report.hospital_name} · reviewing {report.reports.length} screenings
            {report.created_by_email && <> · reviewed by {report.created_by_email}</>}
          </p>
          <p className="text-xs text-ink-400 mt-1">
            Created {new Date(report.created_at).toLocaleString()}
            {report.updated_at !== report.created_at && (
              <> · last edited {new Date(report.updated_at).toLocaleString()}</>
            )}
          </p>
        </Card>

        <Card className="p-6 mb-6">
          <div className="flex items-center justify-between mb-2">
            <p className="font-medium text-ink-900">Reviewer comments</p>
            {!editingComments && (
              <button
                title="Edit comments"
                onClick={() => {
                  setCommentsInput(report.comments);
                  setEditingComments(true);
                }}
                className="text-ink-400 hover:text-brand-600"
              >
                <Pencil size={14} />
              </button>
            )}
          </div>
          {editingComments ? (
            <div>
              <textarea
                className="w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
                rows={4}
                value={commentsInput}
                onChange={(e) => setCommentsInput(e.target.value)}
              />
              <div className="flex items-center gap-2 mt-2">
                <Button size="sm" onClick={saveComments} disabled={saving}>
                  {saving ? "Saving…" : "Save"}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setEditingComments(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-ink-700 whitespace-pre-line">
              {report.comments || <span className="text-ink-400">No comments added yet.</span>}
            </p>
          )}
        </Card>

        {report.reports.map((r, i) => (
          <Card key={r.id} className="p-6 mb-6">
            <div className="flex items-center justify-between flex-wrap gap-2 mb-4">
              <p className="font-medium text-ink-900">
                Screening {i + 1}: {r.session_type === "image" ? "Facial image analysis" : "Questionnaire"}
              </p>
              <div className="flex items-center gap-3">
                {r.risk_category && (
                  <Badge tone={r.risk_category as any}>{r.risk_category} likelihood</Badge>
                )}
                <Link to={`/reports/${r.id}`} className="text-brand-600 text-sm hover:underline">
                  Full report
                </Link>
              </div>
            </div>
            <p className="text-xs text-ink-400 mb-4">
              {new Date(r.created_at).toLocaleString()}
              {r.risk_score !== null && <> · {(r.risk_score * 100).toFixed(0)}% predicted likelihood</>}
            </p>

            {r.summary_stats.length > 0 && (
              <dl className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                {r.summary_stats.map((stat) => (
                  <div key={stat.label} className="bg-ink-50 rounded-lg px-3 py-2">
                    <dt className="text-xs text-ink-400">{stat.label}</dt>
                    <dd className="text-sm font-semibold text-ink-900 whitespace-pre-line">
                      {stat.value}
                    </dd>
                  </div>
                ))}
              </dl>
            )}

            {(r.explanation.length > 0 || r.explanation_note) && (
              <div className="mb-4">
                <p className="text-sm font-medium text-ink-900 mb-2">What influenced this result</p>
                <ExplanationList items={r.explanation} note={r.explanation_note} />
              </div>
            )}

            <ReportFeatureList features={r.features} note={r.note} />
          </Card>
        ))}

        <Card className="p-5 text-sm text-ink-500">
          <div className="flex items-start gap-3">
            <FileCheck2 size={18} className="text-ink-400 shrink-0 mt-0.5" />
            <p>
              This is a combined view of two separately-generated screenings, with added reviewer
              commentary. Neither screening alone, nor this review, is a clinical diagnosis. Please
              consult a qualified healthcare professional for a full evaluation.
            </p>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
