import { useEffect, useState } from "react";
import { KeyRound, RefreshCw, Copy, Check } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { api } from "../../api/client";

interface KeyStatus {
  active: boolean;
  masked_key: string | null;
  created_at: string | null;
}

export default function ApiAccess() {
  const [status, setStatus] = useState<KeyStatus | null>(null);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  function refresh() {
    api.getHospitalApiKeyStatus().then(setStatus);
  }

  useEffect(refresh, []);

  async function handleRegenerate() {
    setBusy(true);
    setNewKey(null);
    try {
      const result = await api.regenerateHospitalApiKey();
      setNewKey(result.api_key);
      refresh();
    } finally {
      setBusy(false);
    }
  }

  function copyKey() {
    if (!newKey) return;
    navigator.clipboard.writeText(newKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <AppShell title="API Access">
      <p className="text-sm text-ink-500 mb-5 max-w-2xl">
        This key authenticates third-party (e.g. hospital EMR) calls to{" "}
        <code className="text-ink-700">/v1/external/*</code>, sent as an{" "}
        <code className="text-ink-700">X-API-Key</code> header. Keep it secret — regenerating
        immediately invalidates the previous key.
      </p>

      <Card className="p-6 max-w-xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center">
            <KeyRound size={20} />
          </div>
          <div>
            <p className="font-medium text-ink-900">Hospital API Key</p>
            {status && (
              <Badge tone={status.active ? "low" : "neutral"}>
                {status.active ? "Active" : "No key issued"}
              </Badge>
            )}
          </div>
        </div>

        {status?.masked_key && (
          <>
            <p className="text-xs text-ink-500 mb-1">
              Current key (hidden for security — only the last 4 characters are shown)
            </p>
            <div className="bg-ink-50 border border-ink-200 rounded-lg px-4 py-3 font-mono text-sm text-ink-700 mb-2">
              {status.masked_key}
            </div>
          </>
        )}
        {status?.created_at && (
          <p className="text-xs text-ink-400 mb-4">
            Issued {new Date(status.created_at).toLocaleString()}
          </p>
        )}

        {newKey && (
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 mb-4">
            <p className="text-xs font-medium text-amber-800 mb-1">
              Your new full API key — copy it now, it won't be shown again:
            </p>
            <div className="flex items-center gap-2">
              <p className="font-mono text-sm text-amber-900 break-all flex-1">{newKey}</p>
              <button
                onClick={copyKey}
                title="Copy to clipboard"
                className="text-amber-700 hover:text-amber-900 shrink-0"
              >
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>
          </div>
        )}

        <Button onClick={handleRegenerate} disabled={busy} variant="secondary">
          <RefreshCw size={16} className={busy ? "animate-spin" : ""} />
          {status?.active ? "Regenerate Key" : "Generate Key"}
        </Button>
      </Card>
    </AppShell>
  );
}
