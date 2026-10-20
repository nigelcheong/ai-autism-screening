import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CalendarClock, Pencil } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { api } from "../../api/client";
import { PersonalInfo as PersonalInfoType, Relation } from "../../types";
import {
  ageInMonths,
  ageLabel,
  instrumentForAgeMonths,
  isEligibleForImageScreening,
  INSTRUMENT_LABELS,
} from "../../utils/age";
import { useDiagnosisFlow } from "./DiagnosisFlowContext";

const RELATIONS: Relation[] = [
  "Self",
  "Parent",
  "Relative",
  "Health care professional",
  "Others",
];

const GENDER_LABELS: Record<string, string> = { male: "Male", female: "Female", others: "Others" };

const emptyForm: PersonalInfoType = {
  age: 0,
  dob: "",
  gender: "male",
  autism: false,
  jaundice: false,
  used_app_before: false,
  country_of_res: "",
  relation: "Self",
};

const inputClass =
  "mt-1 w-full border border-ink-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500";

export default function PersonalInfo() {
  const navigate = useNavigate();
  const { patientId, patientDob, patientKnownInfo, sessionType, personalInfo, setPersonalInfo } =
    useDiagnosisFlow();

  // Everything but dob (which has its own always-shown "on file" banner) is
  // only pre-filled, not forced -- the summary view below is what actually
  // skips re-asking, and "Edit" always re-opens the full form.
  const [form, setForm] = useState<PersonalInfoType>(
    personalInfo ?? { ...emptyForm, dob: patientDob ?? "", ...patientKnownInfo }
  );

  // Known, non-dob fields already on the patient record -- if every one of
  // these is set, show a compact summary instead of asking the full form
  // again. Booleans can legitimately be "No" (false), so check for
  // "recorded at all" rather than truthiness.
  const infoOnRecord =
    !!patientKnownInfo &&
    patientKnownInfo.gender !== undefined &&
    patientKnownInfo.autism !== undefined &&
    patientKnownInfo.jaundice !== undefined &&
    patientKnownInfo.used_app_before !== undefined &&
    !!patientKnownInfo.country_of_res &&
    !!patientKnownInfo.relation;
  const [editingInfo, setEditingInfo] = useState(false);

  useEffect(() => {
    if (!patientId) navigate("/diagnose", { replace: true });
    else if (!sessionType) navigate("/diagnose/type", { replace: true });
  }, [patientId, sessionType, navigate]);

  // Re-sync whenever the selected patient changes, so switching patients
  // (without a full remount, e.g. browser back/forward) can't leave a
  // previous patient's personal info or date of birth showing.
  useEffect(() => {
    setForm(personalInfo ?? { ...emptyForm, dob: patientDob ?? "", ...patientKnownInfo });
    setEditingInfo(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  function update<K extends keyof PersonalInfoType>(key: K, value: PersonalInfoType[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  const months = form.dob ? ageInMonths(form.dob) : null;
  const instrument = months !== null ? instrumentForAgeMonths(months) : null;
  const dobOnRecord = !!patientDob;
  const imageAgeBlocked =
    sessionType === "image" && months !== null && !isEligibleForImageScreening(months);

  async function handleNext() {
    if (months === null || imageAgeBlocked) return;
    // Saved every time (not just the first), so a legacy patient missing
    // some fields gets fully backfilled, and an edit this time sticks for
    // next time too.
    api
      .updatePatient(patientId, {
        dob: dobOnRecord ? undefined : form.dob,
        gender: form.gender,
        autism: form.autism,
        jaundice: form.jaundice,
        used_app_before: form.used_app_before,
        country_of_res: form.country_of_res,
        relation: form.relation,
      })
      .catch(() => {});
    setPersonalInfo({ ...form, age: Math.floor(months / 12) });
    navigate(sessionType === "image" ? "/diagnose/images" : "/diagnose/questionnaire");
  }

  if (!patientId || !sessionType) return null;

  return (
    <AppShell title="New Diagnosis — Personal Information">
      <div className="max-w-lg mx-auto">
        <p className="text-sm text-ink-500 mb-6">
          A few details about the patient before we begin
          {sessionType === "image" ? " the image upload" : ""}.
        </p>

        <Card className="p-6">
          <div className="space-y-4">
            {dobOnRecord ? (
              <div className="flex items-start gap-3 bg-ink-50 border border-ink-200 rounded-lg px-4 py-3">
                <CalendarClock size={18} className="text-ink-500 shrink-0 mt-0.5" />
                <p className="text-sm text-ink-700">
                  Date of birth on file:{" "}
                  <span className="font-medium">
                    {new Date(patientDob!).toLocaleDateString()}
                  </span>
                  {months !== null && <> ({ageLabel(months)} old)</>}
                </p>
              </div>
            ) : (
              <label className="block">
                <span className="text-sm font-medium text-ink-700">Date of birth</span>
                <input
                  type="date"
                  required
                  max={new Date().toISOString().slice(0, 10)}
                  className={inputClass}
                  value={form.dob ?? ""}
                  onChange={(e) => update("dob", e.target.value)}
                />
                <span className="text-xs text-ink-400 mt-1 block">
                  No date of birth on file for this patient yet — this is saved after you continue,
                  so it won't be asked again.
                </span>
              </label>
            )}

            {sessionType === "questionnaire" && months !== null && instrument && (
              <div className="flex items-start gap-3 bg-brand-50 border border-brand-100 rounded-lg px-4 py-3">
                <CalendarClock size={18} className="text-brand-600 shrink-0 mt-0.5" />
                <p className="text-sm text-brand-900">
                  This screening will use{" "}
                  <span className="font-medium">{INSTRUMENT_LABELS[instrument]}</span>.
                </p>
              </div>
            )}

            {imageAgeBlocked && (
              <div className="flex items-start gap-3 bg-red-50 border border-red-100 rounded-lg px-4 py-3">
                <CalendarClock size={18} className="text-red-600 shrink-0 mt-0.5" />
                <p className="text-sm text-red-900">
                  Image screening is validated for children aged 18 months to 11 years. This
                  patient is {ageLabel(months!)} old — please use the questionnaire screening
                  instead.
                </p>
              </div>
            )}

            {infoOnRecord && !editingInfo ? (
              <div className="bg-ink-50 border border-ink-200 rounded-lg px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-ink-700">Other details on file</p>
                  <button
                    onClick={() => setEditingInfo(true)}
                    className="flex items-center gap-1 text-xs text-brand-600 hover:underline"
                  >
                    <Pencil size={12} />
                    Edit
                  </button>
                </div>
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <dt className="text-xs text-ink-400">Gender</dt>
                    <dd className="text-ink-900">{GENDER_LABELS[form.gender] ?? form.gender}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-ink-400">Relation to patient</dt>
                    <dd className="text-ink-900">{form.relation}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-ink-400">Family history of autism</dt>
                    <dd className="text-ink-900">{form.autism ? "Yes" : "No"}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-ink-400">Born with jaundice</dt>
                    <dd className="text-ink-900">{form.jaundice ? "Yes" : "No"}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-ink-400">Used a screening app before</dt>
                    <dd className="text-ink-900">{form.used_app_before ? "Yes" : "No"}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-ink-400">Country of residence</dt>
                    <dd className="text-ink-900">{form.country_of_res}</dd>
                  </div>
                </dl>
              </div>
            ) : (
              <>
                <label className="block">
                  <span className="text-sm font-medium text-ink-700">Gender</span>
                  <select
                    className={inputClass}
                    value={form.gender}
                    onChange={(e) => update("gender", e.target.value as PersonalInfoType["gender"])}
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="others">Others</option>
                  </select>
                </label>

                <YesNo
                  label="Family history of autism?"
                  value={form.autism}
                  onChange={(v) => update("autism", v)}
                />
                <YesNo
                  label="Born with jaundice?"
                  value={form.jaundice}
                  onChange={(v) => update("jaundice", v)}
                />
                <YesNo
                  label="Used a screening app before?"
                  value={form.used_app_before}
                  onChange={(v) => update("used_app_before", v)}
                />

                <label className="block">
                  <span className="text-sm font-medium text-ink-700">Country of residence</span>
                  <input
                    type="text"
                    className={inputClass}
                    value={form.country_of_res}
                    onChange={(e) => update("country_of_res", e.target.value)}
                  />
                </label>

                <label className="block">
                  <span className="text-sm font-medium text-ink-700">Relation to patient</span>
                  <select
                    className={inputClass}
                    value={form.relation}
                    onChange={(e) => update("relation", e.target.value as Relation)}
                  >
                    {RELATIONS.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            )}
          </div>

          <div className="mt-6 flex justify-between">
            <Button variant="ghost" onClick={() => navigate("/diagnose/type")}>
              Back
            </Button>
            <Button disabled={!form.dob || imageAgeBlocked} onClick={handleNext}>
              Next: {sessionType === "image" ? "Upload Images" : "Questionnaire"}
            </Button>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

function YesNo({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div>
      <span className="text-sm font-medium text-ink-700 block mb-1">{label}</span>
      <div className="flex gap-3">
        <Button
          type="button"
          size="sm"
          variant={value ? "primary" : "secondary"}
          onClick={() => onChange(true)}
        >
          Yes
        </Button>
        <Button
          type="button"
          size="sm"
          variant={!value ? "primary" : "secondary"}
          onClick={() => onChange(false)}
        >
          No
        </Button>
      </div>
    </div>
  );
}
