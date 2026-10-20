import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, ShieldCheck, Check } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { LoadingOverlay } from "../../components/ui/LoadingOverlay";
import { DoctorAssignSelect } from "../../components/DoctorAssignSelect";
import { api } from "../../api/client";
import { instrumentForAgeMonths, ageInMonths, INSTRUMENT_LABELS } from "../../utils/age";
import { useAuth } from "../../context/AuthContext";
import { useDiagnosisFlow } from "./DiagnosisFlowContext";

const ADVANCE_DELAY_MS = 350;

interface LiveQuestion {
  id: string;
  key: string;
  text: string;
  options: string[];
}

export default function Questionnaire() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  const { patientId, patientName, personalInfo, doctorId, setDoctorId, setAnswers, setResult } =
    useDiagnosisFlow();
  const [questions, setQuestions] = useState<LiveQuestion[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [responses, setResponses] = useState<Record<string, number>>({});
  const [index, setIndex] = useState(0);
  const [justPicked, setJustPicked] = useState<number | null>(null);
  const [readyToSubmit, setReadyToSubmit] = useState(false);
  const [consent, setConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const instrument = personalInfo?.dob ? instrumentForAgeMonths(ageInMonths(personalInfo.dob)) : null;

  useEffect(() => {
    // A doctor running their own screening is implicitly the assigned
    // doctor -- don't make them pick themselves from a list.
    if (isDoctor && user && !doctorId) setDoctorId(user.subject);
  }, [isDoctor, user, doctorId, setDoctorId]);

  useEffect(() => {
    if (!patientId) {
      navigate("/diagnose", { replace: true });
      return;
    }
    if (!personalInfo || !instrument) {
      navigate("/diagnose/personal-info", { replace: true });
      return;
    }
    api
      .getActiveQuestions(instrument)
      .then(setQuestions)
      .catch((err) => setLoadError((err as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId, personalInfo, navigate]);

  const question = questions?.[index];
  const total = questions?.length ?? 0;
  const progress = total
    ? Math.round(((index + (justPicked !== null || readyToSubmit ? 1 : 0)) / total) * 100)
    : 0;

  async function handleSubmit() {
    if (!consent || !personalInfo || !instrument) return;
    setSubmitting(true);
    setSubmitError(null);
    setAnswers(responses);
    try {
      const result = await api.submitQuestionnaire({
        patient_id: patientId,
        instrument,
        answers: responses,
        personal_info: personalInfo,
        consent_given: consent,
        assigned_doctor_id: doctorId || null,
      });
      setResult(result as any);
      navigate("/diagnose/results");
    } catch (err) {
      setSubmitError((err as Error).message);
      setSubmitting(false);
    }
  }

  function handleAnswer(optionIndex: number) {
    if (!question || !questions) return;
    setJustPicked(optionIndex);
    const next = { ...responses, [question.key]: optionIndex };
    setResponses(next);

    setTimeout(() => {
      setJustPicked(null);
      if (index + 1 < questions.length) {
        setIndex(index + 1);
      } else {
        setReadyToSubmit(true);
      }
    }, ADVANCE_DELAY_MS);
  }

  function handleBack() {
    if (readyToSubmit) {
      setReadyToSubmit(false);
    } else if (index === 0) {
      navigate("/diagnose/personal-info");
    } else {
      setIndex(index - 1);
    }
  }

  if (!patientId || !personalInfo || !instrument) return null;

  if (loadError) {
    return (
      <AppShell title="New Diagnosis — Questionnaire">
        <p className="text-red-600 text-sm max-w-lg mx-auto">{loadError}</p>
      </AppShell>
    );
  }

  if (!questions) {
    return (
      <AppShell title="New Diagnosis — Questionnaire">
        <p className="text-ink-400 text-sm max-w-lg mx-auto">Loading questions…</p>
      </AppShell>
    );
  }

  return (
    <AppShell title={`New Diagnosis — ${INSTRUMENT_LABELS[instrument]}`}>
      <LoadingOverlay
        active={submitting}
        label="Running screening"
        messages={[
          "Scoring your answers…",
          `Running the trained ${INSTRUMENT_LABELS[instrument].split(" ")[0]} model…`,
          "Calculating the risk score…",
          "Preparing the report…",
        ]}
      />
      <div className="max-w-lg mx-auto">
        <div className="flex items-center justify-between mb-2">
          <button
            onClick={handleBack}
            className="flex items-center gap-1 text-sm text-ink-500 hover:text-ink-900"
          >
            <ChevronLeft size={16} />
            Back
          </button>
          <span className="text-sm text-ink-500">
            {readyToSubmit ? "Review" : `${index + 1} / ${total}`}
          </span>
        </div>

        <div className="h-1.5 bg-ink-100 rounded-full overflow-hidden mb-8">
          <div
            className="h-full bg-brand-600 transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {!readyToSubmit && question && (
          <Card key={question.id} className="p-6 sm:p-8 animate-fadeIn">
            <p className="text-lg sm:text-xl font-medium text-ink-900 leading-relaxed text-center min-h-[4.5rem] flex items-center justify-center">
              {question.text}
            </p>

            <div className="mt-6 space-y-2">
              {question.options.map((option, i) => (
                <button
                  key={option}
                  onClick={() => handleAnswer(i)}
                  disabled={justPicked !== null}
                  className={`w-full flex items-center justify-between gap-2 px-4 py-3 rounded-xl border-2 text-left transition-all active:scale-[0.99] ${
                    justPicked === i
                      ? "border-brand-600 bg-brand-50"
                      : "border-ink-200 hover:border-brand-300"
                  }`}
                >
                  <span className="font-medium text-ink-900">{option}</span>
                  {justPicked === i && <Check size={18} className="text-brand-600 shrink-0" />}
                </button>
              ))}
            </div>
          </Card>
        )}

        {readyToSubmit && (
          <Card className="p-6 sm:p-8 animate-fadeIn">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
                <ShieldCheck size={20} />
              </div>
              <div>
                <p className="font-medium text-ink-900">All {total} questions answered</p>
                <p className="text-sm text-ink-500">Ready to run the screening for {patientName}.</p>
              </div>
            </div>

            {!isDoctor && <DoctorAssignSelect value={doctorId} onChange={setDoctorId} />}

            <label className="flex items-start gap-2 mt-5 text-sm text-ink-700 cursor-pointer">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
              />
              I confirm the patient/guardian has given consent to run this autism screening and
              store the results.
            </label>

            {submitError && <p className="text-red-600 text-sm mt-3">{submitError}</p>}

            <Button
              className="w-full mt-5"
              disabled={!consent || submitting}
              onClick={handleSubmit}
            >
              {submitting ? "Running screening…" : "Submit Screening"}
            </Button>
          </Card>
        )}

        <div className="flex justify-center gap-1.5 mt-6">
          {questions.map((q, i) => (
            <div
              key={q.id}
              className={`h-1.5 rounded-full transition-all ${
                !readyToSubmit && i === index
                  ? "w-5 bg-brand-600"
                  : responses[q.key] !== undefined
                  ? "w-1.5 bg-brand-300"
                  : "w-1.5 bg-ink-200"
              }`}
            />
          ))}
        </div>
      </div>
    </AppShell>
  );
}
