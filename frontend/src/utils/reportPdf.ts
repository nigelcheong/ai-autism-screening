import { jsPDF } from "jspdf";

export interface ReportPdfData {
  id: string;
  patient_name: string;
  patient_dob: string | null;
  patient_age_label: string | null;
  patient_external_id: string | null;
  hospital_name: string;
  doctor_email: string | null;
  session_type: string;
  risk_score: number | null;
  risk_category: string | null;
  model_version: string | null;
  model_description: string;
  created_at: string;
  generated_at: string;
  summary_stats: { label: string; value: string }[];
  features: {
    label: string;
    value: string;
    contributed_to_risk: boolean;
    image_data_url: string | null;
    heatmap_data_url: string | null;
    shap_diagram_data_url: string | null;
  }[];
  explanation: { label: string; direction: "increased" | "decreased"; magnitude: number }[];
  explanation_note: string;
  note: string;
}

const MARGIN = 18;
const PAGE_WIDTH = 210;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2;

export function buildReportPdf(report: ReportPdfData): jsPDF {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  let y = MARGIN;

  function ensureSpace(lines: number, lineHeight = 6) {
    if (y + lines * lineHeight > 285) {
      doc.addPage();
      y = MARGIN;
    }
  }

  doc.setFontSize(10);
  doc.setTextColor(120);
  doc.text(report.hospital_name, MARGIN, y);
  y += 7;

  doc.setFontSize(18);
  doc.setTextColor(20);
  doc.setFont("helvetica", "bold");
  doc.text("Autism Screening Report", MARGIN, y);
  y += 3;
  doc.setDrawColor(220);
  doc.line(MARGIN, y, PAGE_WIDTH - MARGIN, y);
  y += 8;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(60);

  const fields: [string, string][] = [
    ["Patient", report.patient_name],
    [
      "Date of birth",
      report.patient_dob
        ? `${new Date(report.patient_dob).toLocaleDateString()}${
            report.patient_age_label ? ` (${report.patient_age_label} old)` : ""
          }`
        : "—",
    ],
    ["Patient ID", report.patient_external_id ?? "—"],
    ["Reviewing clinician", report.doctor_email ?? "Unassigned"],
    ["Screening date", new Date(report.created_at).toLocaleString()],
    ["Screening type", report.session_type === "image" ? "Facial image analysis" : "Questionnaire"],
  ];

  const colWidth = CONTENT_WIDTH / 2;
  fields.forEach(([label, value], i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = MARGIN + col * colWidth;
    const rowY = y + row * 12;
    doc.setTextColor(140);
    doc.setFontSize(8);
    doc.text(label.toUpperCase(), x, rowY);
    doc.setTextColor(20);
    doc.setFontSize(10);
    doc.text(value, x, rowY + 5, { maxWidth: colWidth - 4 });
  });
  y += Math.ceil(fields.length / 2) * 12 + 6;

  ensureSpace(4);
  doc.setFillColor(245, 246, 248);
  doc.rect(MARGIN, y, CONTENT_WIDTH, 20, "F");
  doc.setFontSize(10);
  doc.setTextColor(60);
  doc.text("Screening result", MARGIN + 4, y + 8);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  const resultText = `${report.risk_category ?? "—"} likelihood${
    report.risk_score !== null ? ` (${(report.risk_score * 100).toFixed(0)}%)` : ""
  }`;
  doc.text(resultText, MARGIN + 4, y + 15);
  doc.setFont("helvetica", "normal");
  y += 28;

  if (report.summary_stats.length > 0) {
    ensureSpace(2);
    doc.setFontSize(12);
    doc.setTextColor(20);
    doc.setFont("helvetica", "bold");
    doc.text("Model insights", MARGIN, y);
    y += 7;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    for (const stat of report.summary_stats) {
      ensureSpace(1, 6);
      doc.setTextColor(140);
      doc.text(stat.label, MARGIN, y);
      doc.setTextColor(20);
      doc.setFont("helvetica", "bold");
      doc.text(stat.value, MARGIN + 75, y, { maxWidth: CONTENT_WIDTH - 75 });
      doc.setFont("helvetica", "normal");
      y += 6;
    }
    y += 4;
  }

  if (report.explanation.length > 0) {
    ensureSpace(3);
    doc.setFontSize(12);
    doc.setTextColor(20);
    doc.setFont("helvetica", "bold");
    doc.text("What influenced this result", MARGIN, y);
    y += 7;

    const labelWidth = 38; // reserved for "increased/decreased risk estimate" text after the bar
    const barMaxWidth = CONTENT_WIDTH - labelWidth;
    for (const item of report.explanation) {
      ensureSpace(2, 10);
      doc.setFontSize(9);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(20);
      const lines = doc.splitTextToSize(item.label, CONTENT_WIDTH);
      doc.text(lines, MARGIN, y);
      y += lines.length * 4.5 + 1;

      const barWidth = Math.max(1, (item.magnitude / 100) * barMaxWidth);
      if (item.direction === "increased") {
        doc.setFillColor(217, 119, 6); // amber -- pushed the estimate up
      } else {
        doc.setFillColor(5, 150, 105); // green -- pushed the estimate down
      }
      doc.rect(MARGIN, y, barWidth, 3, "F");
      doc.setFontSize(7.5);
      doc.setTextColor(130);
      doc.text(`${item.direction} risk estimate`, MARGIN + barMaxWidth + 3, y + 2.5);
      y += 7;
    }

    if (report.explanation_note) {
      doc.setFontSize(8);
      const noteLines = doc.splitTextToSize(report.explanation_note, CONTENT_WIDTH);
      ensureSpace(noteLines.length, 4);
      doc.setTextColor(130);
      doc.text(noteLines, MARGIN, y);
      y += noteLines.length * 4 + 6;
    } else {
      y += 4;
    }
  }

  ensureSpace(2);
  doc.setFontSize(12);
  doc.setTextColor(20);
  doc.setFont("helvetica", "bold");
  doc.text("Features considered", MARGIN, y);
  y += 7;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);

  for (const f of report.features) {
    if (f.image_data_url || f.heatmap_data_url) {
      ensureSpace(4, 6);
      const imgSize = 22;
      let x = MARGIN;
      for (const src of [f.image_data_url, f.heatmap_data_url]) {
        if (!src) continue;
        try {
          doc.addImage(src, "JPEG", x, y, imgSize, imgSize);
        } catch {
          // malformed data URL -- skip the thumbnail rather than fail the whole PDF
        }
        x += imgSize + 2;
      }
      doc.setTextColor(20);
      const text = `${f.label}: ${f.value}`;
      const lines = doc.splitTextToSize(text, CONTENT_WIDTH - (x - MARGIN) - 4);
      doc.text(lines, x + 2, y + 5);
      y += imgSize + 3;

      if (f.shap_diagram_data_url) {
        const diagramSize = Math.min(CONTENT_WIDTH, 110);
        ensureSpace(diagramSize / 6 + 2);
        doc.setFontSize(8);
        doc.setTextColor(130);
        doc.text(`SHAP contribution by region (${f.label})`, MARGIN, y);
        y += 4;
        try {
          doc.addImage(f.shap_diagram_data_url, "JPEG", MARGIN, y, diagramSize, diagramSize);
        } catch {
          // malformed data URL -- skip rather than fail the whole PDF
        }
        y += diagramSize + 4;
        doc.setFontSize(9.5);
      }
      continue;
    }
    ensureSpace(1, 6);
    doc.setTextColor(f.contributed_to_risk ? 180 : 150, f.contributed_to_risk ? 60 : 150, 60);
    doc.text(f.contributed_to_risk ? "[x]" : "[ ]", MARGIN, y);
    doc.setTextColor(20);
    const text = `${f.label}: ${f.value}`;
    const lines = doc.splitTextToSize(text, CONTENT_WIDTH - 10);
    doc.text(lines, MARGIN + 9, y);
    y += lines.length * 5;
  }
  y += 4;

  if (report.note) {
    doc.setFontSize(8.5);
    const noteLines = doc.splitTextToSize(report.note, CONTENT_WIDTH);
    ensureSpace(noteLines.length, 4.5);
    doc.setTextColor(130);
    doc.text(noteLines, MARGIN, y);
    y += noteLines.length * 4.5 + 6;
  }

  ensureSpace(5);
  doc.setDrawColor(220);
  doc.line(MARGIN, y, PAGE_WIDTH - MARGIN, y);
  y += 7;
  doc.setFontSize(9.5);
  doc.setTextColor(60);
  doc.setFont("helvetica", "bold");
  doc.text(`Scored by: ${report.model_description}`, MARGIN, y, { maxWidth: CONTENT_WIDTH });
  y += 6;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(130);
  doc.text(
    `Model reference: ${report.model_version ?? "—"}  ·  Report generated ${new Date(
      report.generated_at
    ).toLocaleString()}`,
    MARGIN,
    y
  );
  y += 6;
  const disclaimer =
    "This is a screening aid, not a clinical diagnosis. Please consult a qualified healthcare professional for a full evaluation.";
  doc.text(doc.splitTextToSize(disclaimer, CONTENT_WIDTH), MARGIN, y);

  return doc;
}

export function downloadReportPdf(report: ReportPdfData) {
  const doc = buildReportPdf(report);
  doc.save(`report-${report.patient_name.replace(/\s+/g, "_")}-${report.id.slice(0, 8)}.pdf`);
}
