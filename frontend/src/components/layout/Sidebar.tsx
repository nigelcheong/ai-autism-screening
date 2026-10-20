import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Users,
  FilePlus2,
  KeyRound,
  Building2,
  ListChecks,
  LogOut,
  Stethoscope,
  ClipboardList,
  UserCog,
  X,
  BarChart3,
  History,
} from "lucide-react";
import { useAuth } from "../../context/AuthContext";

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
}

const STAFF_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/patients", label: "Patients", icon: Users },
  { to: "/diagnose", label: "New Diagnosis", icon: FilePlus2 },
];

const DOCTOR_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/cases", label: "My Cases", icon: ClipboardList },
  { to: "/diagnose", label: "New Diagnosis", icon: FilePlus2 },
];

const HOSPITAL_ADMIN_ITEMS: NavItem[] = [
  { to: "/admin/api-access", label: "API Access", icon: KeyRound },
];

const PLATFORM_ADMIN_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/platform/usage", label: "Usage", icon: BarChart3 },
  { to: "/platform/hospitals", label: "Hospitals", icon: Building2 },
  { to: "/platform/questions", label: "Question Bank", icon: ListChecks },
  { to: "/platform/audit-log", label: "Audit Log", icon: History },
];

const ACCOUNT_ITEM: NavItem = { to: "/account", label: "Account", icon: UserCog };

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export function Sidebar({ open, onClose }: SidebarProps) {
  const { user, logout } = useAuth();
  const items = [
    ...(user?.role === "platform_admin"
      ? PLATFORM_ADMIN_ITEMS
      : user?.role === "hospital_admin"
      ? [...STAFF_ITEMS, ...HOSPITAL_ADMIN_ITEMS]
      : user?.role === "doctor"
      ? DOCTOR_ITEMS
      : STAFF_ITEMS),
    ACCOUNT_ITEM,
  ];

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 bg-black/30 z-30 sm:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={`w-64 sm:w-60 shrink-0 bg-white border-r border-ink-200 flex flex-col h-screen fixed sm:sticky top-0 z-40 transition-transform duration-200 ${
          open ? "translate-x-0" : "-translate-x-full sm:translate-x-0"
        }`}
      >
        <div className="px-5 py-5 flex items-center justify-between border-b border-ink-200">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-brand-600 text-white flex items-center justify-center">
              <Stethoscope size={18} />
            </div>
            <span className="font-semibold text-ink-900">Autism Screen</span>
          </div>
          <button onClick={onClose} className="sm:hidden text-ink-400 hover:text-ink-700">
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-brand-50 text-brand-700"
                    : "text-ink-600 hover:bg-ink-100 hover:text-ink-900"
                }`
              }
            >
              <item.icon size={18} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="px-3 py-4 border-t border-ink-200">
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-ink-600 hover:bg-ink-100"
          >
            <LogOut size={18} />
            Log out
          </button>
        </div>
      </aside>
    </>
  );
}
