import { useState } from "react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { PasswordInput } from "../../components/ui/PasswordInput";
import { api } from "../../api/client";
import { useAuth } from "../../context/AuthContext";

export default function ChangePassword() {
  const { user } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(false);
    if (next.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    if (next !== confirm) {
      setError("New password and confirmation don't match.");
      return;
    }
    setSubmitting(true);
    try {
      await api.changePassword(current, next);
      setCurrent("");
      setNext("");
      setConfirm("");
      setSuccess(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppShell title="Account">
      <div className="max-w-sm">
        <Card className="p-6">
          <p className="font-medium text-ink-900 mb-1">Change password</p>
          {user?.email && <p className="text-sm text-ink-500 mb-4">{user.email}</p>}

          <form onSubmit={handleSubmit} className="space-y-3">
            <label className="block">
              <span className="text-sm font-medium text-ink-700">Current password</span>
              <div className="mt-1">
                <PasswordInput
                  autoComplete="current-password"
                  value={current}
                  onChange={(e) => setCurrent(e.target.value)}
                  required
                />
              </div>
            </label>
            <label className="block">
              <span className="text-sm font-medium text-ink-700">New password</span>
              <div className="mt-1">
                <PasswordInput
                  autoComplete="new-password"
                  value={next}
                  onChange={(e) => setNext(e.target.value)}
                  required
                />
              </div>
            </label>
            <label className="block">
              <span className="text-sm font-medium text-ink-700">Confirm new password</span>
              <div className="mt-1">
                <PasswordInput
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                />
              </div>
            </label>

            {error && <p className="text-red-600 text-sm">{error}</p>}
            {success && <p className="text-emerald-600 text-sm">Password updated.</p>}

            <Button type="submit" disabled={submitting} className="w-full">
              {submitting ? "Updating…" : "Update Password"}
            </Button>
          </form>
        </Card>
      </div>
    </AppShell>
  );
}
