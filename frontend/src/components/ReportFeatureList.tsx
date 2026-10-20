import { Check, X } from "lucide-react";

interface Feature {
  label: string;
  value: string;
  contributed_to_risk: boolean;
  image_data_url?: string | null;
  heatmap_data_url?: string | null;
  shap_diagram_data_url?: string | null;
}

export function ReportFeatureList({ features, note }: { features: Feature[]; note?: string }) {
  return (
    <div>
      <ul className="divide-y divide-ink-100">
        {features.map((f, i) => (
          <li key={i} className="py-3 text-left">
            <div className="flex items-start gap-3">
              {f.contributed_to_risk ? (
                <Check size={16} className="text-risk-high mt-0.5 shrink-0" />
              ) : (
                <X size={16} className="text-ink-300 mt-0.5 shrink-0" />
              )}
              {(f.image_data_url || f.heatmap_data_url) && (
                <div className="flex gap-1 shrink-0">
                  {f.image_data_url && (
                    <img
                      src={f.image_data_url}
                      alt={`${f.label} as submitted`}
                      title="As submitted"
                      className="w-16 h-16 rounded-lg object-cover border border-ink-200"
                    />
                  )}
                  {f.heatmap_data_url && (
                    <img
                      src={f.heatmap_data_url}
                      alt={`${f.label} Grad-CAM attention map`}
                      title="Grad-CAM attention map"
                      className="w-16 h-16 rounded-lg object-cover border border-ink-200"
                    />
                  )}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-ink-900">{f.label}</p>
                <p className="text-xs text-ink-500">{f.value}</p>
              </div>
            </div>
            {f.shap_diagram_data_url && (
              <div className="mt-3 ml-7">
                <p className="text-xs text-ink-400 mb-1.5">
                  SHAP contribution by region ({f.label})
                </p>
                <img
                  src={f.shap_diagram_data_url}
                  alt={`${f.label} SHAP region diagram`}
                  className="w-full max-w-sm rounded-lg border border-ink-200"
                />
              </div>
            )}
          </li>
        ))}
        {features.length === 0 && (
          <li className="py-6 text-center text-sm text-ink-400">
            No feature detail available for this session.
          </li>
        )}
      </ul>
      {note && <p className="text-xs text-ink-400 mt-4 text-left">{note}</p>}
    </div>
  );
}
