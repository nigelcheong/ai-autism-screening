export type Instrument = "aq10" | "qchat10";

export function ageInMonths(dobIso: string, asOf: Date = new Date()): number {
  const dob = new Date(dobIso);
  let months = (asOf.getFullYear() - dob.getFullYear()) * 12 + (asOf.getMonth() - dob.getMonth());
  if (asOf.getDate() < dob.getDate()) months -= 1;
  return Math.max(months, 0);
}

export function instrumentForAgeMonths(months: number): Instrument {
  return months < 48 ? "qchat10" : "aq10";
}

export function instrumentForDob(dobIso: string): Instrument {
  return instrumentForAgeMonths(ageInMonths(dobIso));
}

// Image screening covers toddlers through pre-teens: 18 months to 11 years
// (inclusive) -- i.e. below 12 years.
export const IMAGE_SCREENING_MIN_MONTHS = 18;
export const IMAGE_SCREENING_MAX_MONTHS = 144;

export function isEligibleForImageScreening(months: number): boolean {
  return months >= IMAGE_SCREENING_MIN_MONTHS && months < IMAGE_SCREENING_MAX_MONTHS;
}

export function ageLabel(months: number): string {
  const years = Math.floor(months / 12);
  const remMonths = months % 12;
  if (years === 0) return `${remMonths} month${remMonths === 1 ? "" : "s"}`;
  if (remMonths === 0) return `${years} year${years === 1 ? "" : "s"}`;
  return `${years}y ${remMonths}m`;
}

export const INSTRUMENT_LABELS: Record<Instrument, string> = {
  aq10: "AQ-10 (ages 4-11)",
  qchat10: "Q-CHAT-10 (ages 18-48 months)",
};
