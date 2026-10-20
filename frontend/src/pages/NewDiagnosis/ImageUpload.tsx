import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, ChevronLeft, X } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { LoadingOverlay } from "../../components/ui/LoadingOverlay";
import { api } from "../../api/client";
import { DoctorAssignSelect } from "../../components/DoctorAssignSelect";
import { useAuth } from "../../context/AuthContext";
import { useDiagnosisFlow } from "./DiagnosisFlowContext";

const MAX_IMAGES = 10;

export default function ImageUpload() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  const { patientId, personalInfo, doctorId, setDoctorId, setResult } = useDiagnosisFlow();
  const [files, setFiles] = useState<File[]>([]);
  const [consent, setConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [stage, setStage] = useState<"uploading" | "analyzing">("uploading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!patientId) {
      navigate("/diagnose", { replace: true });
    } else if (!personalInfo) {
      navigate("/diagnose/personal-info", { replace: true });
    }
  }, [patientId, personalInfo, navigate]);

  useEffect(() => {
    // A doctor running their own screening is implicitly the assigned
    // doctor -- don't make them pick themselves from a list.
    if (isDoctor && user && !doctorId) setDoctorId(user.subject);
  }, [isDoctor, user, doctorId, setDoctorId]);

  const previews = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => {
    return () => previews.forEach((url) => URL.revokeObjectURL(url));
  }, [previews]);

  function addFiles(incoming: File[]) {
    setFiles((prev) => [...prev, ...incoming].slice(0, MAX_IMAGES));
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  function handleDrop(e: React.DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    addFiles(Array.from(e.dataTransfer.files).slice(0, MAX_IMAGES - files.length));
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) addFiles(Array.from(e.target.files).slice(0, MAX_IMAGES - files.length));
    e.target.value = "";
  }

  async function handleSubmit() {
    if (!consent) return;
    setSubmitting(true);
    setStage("uploading");
    setError(null);
    try {
      const { image_urls } = await api.uploadImages(files);
      setStage("analyzing");
      const result = await api.submitImages({
        patient_id: patientId,
        image_urls,
        personal_info: personalInfo,
        consent_given: consent,
        assigned_doctor_id: doctorId || null,
      });
      setResult(result as any);
      navigate("/diagnose/results");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  if (!patientId || !personalInfo) return null;

  return (
    <AppShell title="New Diagnosis — Upload Images">
      <LoadingOverlay
        active={submitting}
        label={stage === "uploading" ? "Uploading images" : "Analyzing images"}
        messages={
          stage === "uploading"
            ? ["Uploading photos securely…", "Validating image files…"]
            : [
                "Running the DenseNet-201 CNN model…",
                "Scoring each photo…",
                "Computing Grad-CAM and SHAP explanations (this takes a bit longer)…",
                "Combining results across images…",
                "Preparing the report…",
              ]
        }
      />
      <div className="max-w-lg mx-auto">
        <button
          onClick={() => navigate("/diagnose/personal-info")}
          className="flex items-center gap-1 text-sm text-ink-500 hover:text-ink-900 mb-3"
        >
          <ChevronLeft size={16} />
          Back
        </button>

        <Card className="p-6">
          <label
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="border-2 border-dashed border-ink-200 rounded-xl p-10 text-center text-ink-400 flex flex-col items-center gap-2 cursor-pointer hover:border-ink-300"
          >
            <UploadCloud size={28} className="text-ink-300" />
            Drag and drop or click to select up to {MAX_IMAGES} images
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              className="hidden"
              onChange={handleFileInput}
            />
          </label>

          {files.length > 0 && (
            <div className="mt-4 grid grid-cols-3 sm:grid-cols-4 gap-3">
              {files.map((f, i) => (
                <div key={`${f.name}-${i}`} className="relative group">
                  <img
                    src={previews[i]}
                    alt={f.name}
                    className="w-full h-24 object-cover rounded-lg border border-ink-200"
                  />
                  <button
                    type="button"
                    onClick={() => removeFile(i)}
                    disabled={submitting}
                    title="Remove"
                    className="absolute -top-2 -right-2 bg-white border border-ink-200 rounded-full p-0.5 text-ink-500 hover:text-red-600 shadow-sm"
                  >
                    <X size={14} />
                  </button>
                  <p className="text-xs text-ink-400 mt-1 truncate">{f.name}</p>
                </div>
              ))}
            </div>
          )}

          {!isDoctor && (
            <div className="mt-4">
              <DoctorAssignSelect value={doctorId} onChange={setDoctorId} />
            </div>
          )}

          <label className="flex items-start gap-2 mt-4 text-sm text-ink-700 cursor-pointer">
            <input
              type="checkbox"
              className="mt-0.5"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
            />
            I confirm the patient/guardian has given consent to run this autism screening and
            store the results.
          </label>

          {error && <p className="text-red-600 text-sm mt-4">{error}</p>}

          <div className="mt-6 flex justify-end">
            <Button disabled={submitting || files.length === 0 || !consent} onClick={handleSubmit}>
              {submitting ? "Running screening…" : "Submit"}
            </Button>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
