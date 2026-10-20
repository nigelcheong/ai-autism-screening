import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { setUnauthorizedHandler } from "../api/client";
import { Button } from "./ui/Button";

// How long before the token's own `exp` to start warning the user.
const WARNING_LEAD_MS = 2 * 60 * 1000;

/**
 * Keeps the frontend's session in sync with the backend JWT's real expiry:
 * - Any API call that comes back 401 (token expired/invalidated) triggers an
 *   immediate logout + redirect, instead of leaving a confusing inline error.
 * - A countdown modal warns the user a couple of minutes before their token
 *   actually expires, then auto-logs-out exactly at expiry if they haven't
 *   already signed out themselves (there's no refresh-token endpoint, so
 *   the session cannot be silently extended -- re-login is required).
 */
export function SessionWatcher() {
  const { user, isAuthenticated, logout, extendSession } = useAuth();
  const navigate = useNavigate();
  const [secondsLeft, setSecondsLeft] = useState<number | null>(null);
  const [extending, setExtending] = useState(false);
  const [extendError, setExtendError] = useState<string | null>(null);

  function expireNow() {
    logout();
    navigate("/login?expired=1", { replace: true });
  }

  async function stayLoggedIn() {
    setExtending(true);
    setExtendError(null);
    try {
      await extendSession();
      setSecondsLeft(null);
    } catch {
      // The token may have expired in the gap between the warning firing
      // and the user clicking -- fall back to the normal expiry flow.
      setExtendError("Couldn't extend your session — please log in again.");
    } finally {
      setExtending(false);
    }
  }

  useEffect(() => {
    setUnauthorizedHandler(expireNow);
    return () => setUnauthorizedHandler(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!isAuthenticated || !user?.expiresAt) {
      setSecondsLeft(null);
      return;
    }
    const expiresAt = user.expiresAt;

    function tick() {
      const remaining = expiresAt - Date.now();
      if (remaining <= 0) {
        expireNow();
        return;
      }
      setSecondsLeft(remaining <= WARNING_LEAD_MS ? Math.ceil(remaining / 1000) : null);
    }

    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, user?.expiresAt]);

  if (secondsLeft === null) return null;

  const mm = Math.floor(secondsLeft / 60);
  const ss = (secondsLeft % 60).toString().padStart(2, "0");

  return (
    <div className="fixed inset-0 bg-ink-900/50 backdrop-blur-sm flex items-center justify-center z-[60]">
      <div className="bg-white rounded-2xl shadow-xl px-8 py-7 max-w-sm text-center">
        <AlertTriangle size={32} className="text-amber-500 mx-auto mb-3" />
        <p className="font-semibold text-ink-900">Your session is about to expire</p>
        <p className="text-sm text-ink-500 mt-2">
          You'll be signed out automatically in{" "}
          <span className="font-mono font-medium text-ink-700">
            {mm}:{ss}
          </span>
          .
        </p>
        {extendError && <p className="text-red-600 text-sm mt-3">{extendError}</p>}
        <div className="flex flex-col gap-2 mt-5">
          <Button onClick={stayLoggedIn} disabled={extending}>
            {extending ? "Extending…" : "Stay logged in"}
          </Button>
          <Button variant="secondary" onClick={expireNow} disabled={extending}>
            Log out now
          </Button>
        </div>
      </div>
    </div>
  );
}
