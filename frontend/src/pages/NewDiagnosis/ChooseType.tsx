import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ClipboardCheck, ImageIcon } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { useDiagnosisFlow } from "./DiagnosisFlowContext";

export default function ChooseType() {
  const navigate = useNavigate();
  const { patientId, patientName, setSessionType } = useDiagnosisFlow();

  useEffect(() => {
    if (!patientId) navigate("/diagnose", { replace: true });
  }, [patientId, navigate]);

  if (!patientId) return null;

  function choose(type: "questionnaire" | "image") {
    setSessionType(type);
    navigate("/diagnose/personal-info");
  }

  return (
    <AppShell title="New Diagnosis">
      <div className="max-w-2xl mx-auto mt-6">
        <div className="flex items-center justify-center gap-2 text-sm text-ink-500 mb-8">
          <span>
            Patient: <span className="font-medium text-ink-900">{patientName}</span>
          </span>
          <button
            onClick={() => navigate("/diagnose")}
            className="text-brand-600 hover:underline"
          >
            Change
          </button>
        </div>
        <div className="grid sm:grid-cols-2 gap-5">
          <Card
            className="p-8 text-center cursor-pointer hover:border-brand-300 hover:shadow-none transition-colors"
            onClick={() => choose("questionnaire")}
          >
            <div className="w-12 h-12 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center mx-auto mb-4">
              <ClipboardCheck size={24} />
            </div>
            <h3 className="font-semibold text-ink-900 mb-1">Run Questionnaire</h3>
            <p className="text-sm text-ink-500">AQ-10 questions + personal information</p>
          </Card>
          <Card
            className="p-8 text-center cursor-pointer hover:border-brand-300 hover:shadow-none transition-colors"
            onClick={() => choose("image")}
          >
            <div className="w-12 h-12 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-4">
              <ImageIcon size={24} />
            </div>
            <h3 className="font-semibold text-ink-900 mb-1">Upload Images</h3>
            <p className="text-sm text-ink-500">Drag and drop a batch of patient images</p>
            <p className="text-xs text-ink-400 mt-1">Ages 18 months to 11 years</p>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
