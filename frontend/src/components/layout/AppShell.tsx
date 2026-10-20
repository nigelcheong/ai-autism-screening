import { ReactNode, useState } from "react";
import { Eye, Menu } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { useAuth } from "../../context/AuthContext";

const ROLE_LABELS: Record<string, string> = {
  platform_admin: "Platform Admin",
  hospital_admin: "Hospital Admin",
  doctor: "Doctor",
  staff: "Staff",
};

export function AppShell({ title, children }: { title: string; children: ReactNode }) {
  const { user, isImpersonating, exitImpersonation } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-ink-50">
      <div className="no-print contents">
        <Sidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />
      </div>
      <div className="flex-1 min-w-0">
        {isImpersonating && (
          <div className="no-print min-h-10 bg-amber-500 text-white text-xs sm:text-sm flex flex-wrap items-center justify-center gap-2 px-4 py-1.5">
            <Eye size={14} />
            <span>
              Viewing as {user?.email ?? "another user"} ({ROLE_LABELS[user?.role ?? ""] ?? user?.role})
            </span>
            <button onClick={exitImpersonation} className="underline font-medium">
              Exit impersonation
            </button>
          </div>
        )}
        <header className="no-print h-16 bg-white border-b border-ink-200 flex items-center justify-between px-4 sm:px-8">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setMobileOpen(true)}
              className="sm:hidden text-ink-500 hover:text-ink-900 shrink-0"
            >
              <Menu size={22} />
            </button>
            <h1 className="text-lg font-semibold text-ink-900 truncate">{title}</h1>
          </div>
          {user && (
            <span className="text-sm text-ink-500 hidden sm:inline shrink-0">
              {ROLE_LABELS[user.role] ?? user.role}
            </span>
          )}
        </header>
        <main className="p-4 sm:p-8 max-w-full overflow-x-hidden">{children}</main>
      </div>
    </div>
  );
}
