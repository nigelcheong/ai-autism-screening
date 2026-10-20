export type Relation =
  | "Self"
  | "Parent"
  | "Relative"
  | "Health care professional"
  | "Others";

export interface PersonalInfo {
  age: number;
  dob?: string; // ISO date; used to pick the screening instrument by age
  gender: "male" | "female" | "others";
  autism: boolean; // family history of autism
  jaundice: boolean;
  used_app_before: boolean;
  country_of_res: string;
  relation: Relation;
}

export interface QuestionnaireSubmission {
  patient_id: string;
  instrument: "aq10" | "qchat10";
  answers: Record<string, number>; // {"aq1": 2, "qc3": 0, ...} -> selected option index
  personal_info: PersonalInfo;
  consent_given: boolean;
}

export interface ImageSubmission {
  patient_id: string;
  image_urls: string[];
  consent_given: boolean;
}

export type RiskCategory = "low" | "moderate" | "high";

export interface DiagnosisResult {
  id: string;
  session_type: "questionnaire" | "image";
  risk_score: number;
  risk_category: RiskCategory;
  model_version: string;
  created_at: string;
}
