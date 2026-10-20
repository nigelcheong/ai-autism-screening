import { createContext, useContext, useState, ReactNode } from "react";
import { PersonalInfo, DiagnosisResult } from "../../types";
import { PatientListItem } from "../../api/client";

export type SessionType = "questionnaire" | "image";

// Demographic/history fields already on the patient's record from a past
// screening, keyed the same as PersonalInfo, so PersonalInfo.tsx can pre-fill
// instead of asking again.
export type KnownPatientInfo = Partial<
  Pick<PersonalInfo, "gender" | "autism" | "jaundice" | "used_app_before" | "country_of_res" | "relation">
>;

interface DiagnosisFlowState {
  patientId: string;
  patientName: string;
  patientDob: string | null;
  patientKnownInfo: KnownPatientInfo | null;
  sessionType: SessionType | null;
  answers: Record<string, number>;
  personalInfo: PersonalInfo | null;
  doctorId: string;
  result: DiagnosisResult | null;
  selectPatient: (patient: PatientListItem) => void;
  setSessionType: (type: SessionType) => void;
  setAnswers: (answers: Record<string, number>) => void;
  setPersonalInfo: (info: PersonalInfo) => void;
  setDoctorId: (id: string) => void;
  setResult: (result: DiagnosisResult) => void;
  reset: () => void;
}

const DiagnosisFlowContext = createContext<DiagnosisFlowState | undefined>(undefined);

export function DiagnosisFlowProvider({ children }: { children: ReactNode }) {
  const [patientId, setPatientId] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientDob, setPatientDob] = useState<string | null>(null);
  const [patientKnownInfo, setPatientKnownInfo] = useState<KnownPatientInfo | null>(null);
  const [sessionType, setSessionType] = useState<SessionType | null>(null);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [personalInfo, setPersonalInfo] = useState<PersonalInfo | null>(null);
  const [doctorId, setDoctorId] = useState("");
  const [result, setResult] = useState<DiagnosisResult | null>(null);

  function selectPatient(patient: PatientListItem) {
    setPatientId(patient.id);
    setPatientName(patient.name);
    setPatientDob(patient.dob ?? null);
    // Only include keys the patient actually has a value for -- spreading
    // this into form defaults later must not be able to clobber a default
    // with an explicit `undefined` (object spread copies a key's presence
    // even when its value is undefined, unlike `??`).
    const known: KnownPatientInfo = {};
    if (patient.gender != null) known.gender = patient.gender as PersonalInfo["gender"];
    if (patient.autism != null) known.autism = patient.autism;
    if (patient.jaundice != null) known.jaundice = patient.jaundice;
    if (patient.used_app_before != null) known.used_app_before = patient.used_app_before;
    if (patient.country_of_res != null) known.country_of_res = patient.country_of_res;
    if (patient.relation != null) known.relation = patient.relation as PersonalInfo["relation"];
    setPatientKnownInfo(known);
    // Starting a (possibly different) patient's diagnosis -- clear anything
    // left over from a previous patient's flow (personal info, in-progress
    // answers, doctor assignment, prior result), so PersonalInfo doesn't
    // reinitialize its form from a stale personalInfo belonging to someone
    // else.
    setSessionType(null);
    setAnswers({});
    setPersonalInfo(null);
    setDoctorId("");
    setResult(null);
  }

  function reset() {
    setPatientId("");
    setPatientName("");
    setPatientDob(null);
    setPatientKnownInfo(null);
    setSessionType(null);
    setAnswers({});
    setPersonalInfo(null);
    setDoctorId("");
    setResult(null);
  }

  return (
    <DiagnosisFlowContext.Provider
      value={{
        patientId,
        patientName,
        patientDob,
        patientKnownInfo,
        sessionType,
        answers,
        personalInfo,
        doctorId,
        result,
        selectPatient,
        setSessionType,
        setAnswers,
        setPersonalInfo,
        setDoctorId,
        setResult,
        reset,
      }}
    >
      {children}
    </DiagnosisFlowContext.Provider>
  );
}

export function useDiagnosisFlow(): DiagnosisFlowState {
  const ctx = useContext(DiagnosisFlowContext);
  if (!ctx) throw new Error("useDiagnosisFlow must be used within DiagnosisFlowProvider");
  return ctx;
}
