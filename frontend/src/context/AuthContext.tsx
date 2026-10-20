import { createContext, useContext, useState, ReactNode } from "react";
import { api } from "../api/client";

export type Role = "platform_admin" | "hospital_admin" | "doctor" | "staff";

interface AuthUser {
  role: Role;
  hospitalId: string | null;
  subject: string;
  email: string | null;
  impersonatedBy: string | null;
  /** JWT `exp` claim in epoch milliseconds, or null if the token has none. */
  expiresAt: number | null;
}

interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isImpersonating: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  impersonate: (token: string) => void;
  exitImpersonation: () => void;
  /** Reissues the current token with a fresh expiry ("stay logged in"). */
  extendSession: () => Promise<void>;
}

const AuthContext = createContext<AuthState | undefined>(undefined);
const ORIGINAL_TOKEN_KEY = "original_admin_token";

function decodeUser(token: string): AuthUser {
  const payload = JSON.parse(atob(token.split(".")[1]));
  return {
    role: payload.role,
    hospitalId: payload.hospital_id ?? null,
    subject: payload.sub,
    email: payload.email ?? null,
    impersonatedBy: payload.impersonated_by ?? null,
    expiresAt: typeof payload.exp === "number" ? payload.exp * 1000 : null,
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const token = localStorage.getItem("access_token");
    try {
      return token ? decodeUser(token) : null;
    } catch {
      return null;
    }
  });

  async function login(email: string, password: string) {
    const { access_token } = await api.login(email, password);
    localStorage.setItem("access_token", access_token);
    sessionStorage.removeItem(ORIGINAL_TOKEN_KEY);
    setUser(decodeUser(access_token));
  }

  function logout() {
    // fire-and-forget: record the logout before the token that authorizes it is gone
    api.logout().catch(() => {});
    localStorage.removeItem("access_token");
    sessionStorage.removeItem(ORIGINAL_TOKEN_KEY);
    setUser(null);
  }

  function impersonate(token: string) {
    const current = localStorage.getItem("access_token");
    if (current) sessionStorage.setItem(ORIGINAL_TOKEN_KEY, current);
    localStorage.setItem("access_token", token);
    setUser(decodeUser(token));
  }

  async function extendSession() {
    const { access_token } = await api.refreshSession();
    localStorage.setItem("access_token", access_token);
    setUser(decodeUser(access_token));
  }

  function exitImpersonation() {
    const original = sessionStorage.getItem(ORIGINAL_TOKEN_KEY);
    if (!original) return;
    localStorage.setItem("access_token", original);
    sessionStorage.removeItem(ORIGINAL_TOKEN_KEY);
    setUser(decodeUser(original));
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isImpersonating: !!sessionStorage.getItem(ORIGINAL_TOKEN_KEY),
        login,
        logout,
        impersonate,
        exitImpersonation,
        extendSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
