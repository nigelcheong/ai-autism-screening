import { Link } from "react-router-dom";
import { Stethoscope, ClipboardCheck, ImageIcon, ShieldCheck } from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

const FEATURES = [
  {
    icon: ClipboardCheck,
    title: "AQ-10 Questionnaire",
    description:
      "A fast, standardized 10-question screening flow with structured demographic intake.",
  },
  {
    icon: ImageIcon,
    title: "Image-Based Screening",
    description: "Upload a small batch of patient images for model-assisted risk assessment.",
  },
  {
    icon: ShieldCheck,
    title: "Multi-Tenant & Secure",
    description:
      "Hospital data is isolated per tenant, with row-level security and role-based access.",
  },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-ink-50">
      <header className="border-b border-ink-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-brand-600 text-white flex items-center justify-center">
              <Stethoscope size={18} />
            </div>
            <span className="font-semibold text-ink-900">Autism Screen</span>
          </div>
          <Link to="/login">
            <Button variant="secondary">Staff Login</Button>
          </Link>
        </div>
      </header>

      <section className="max-w-4xl mx-auto px-6 pt-24 pb-16 text-center">
        <h1 className="text-4xl sm:text-5xl font-bold text-ink-900 tracking-tight">
          Autism screening,
          <br />
          <span className="text-brand-600">built for hospitals.</span>
        </h1>
        <p className="mt-5 text-lg text-ink-500 max-w-2xl mx-auto">
          Run questionnaire and image-based screenings, track results across your patients, and
          give your clinical staff a single dashboard for every case.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          <Link to="/login">
            <Button size="lg">Get Started</Button>
          </Link>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-24 grid sm:grid-cols-3 gap-5">
        {FEATURES.map((f) => (
          <Card key={f.title} className="p-6">
            <div className="w-10 h-10 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center mb-4">
              <f.icon size={20} />
            </div>
            <h3 className="font-semibold text-ink-900 mb-1">{f.title}</h3>
            <p className="text-sm text-ink-500">{f.description}</p>
          </Card>
        ))}
      </section>
    </div>
  );
}
