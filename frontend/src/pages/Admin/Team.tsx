import { Fragment, useEffect, useState } from "react";
import { Plus, Trash2, Pencil, X } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { PasswordInput } from "../../components/ui/PasswordInput";
import { api } from "../../api/client";

interface HospitalUser {
  id: string;
  email: string;
  role: string;
}

const ROLES = [
  { value: "staff", label: "Staff" },
  { value: "doctor", label: "Doctor" },
];

const inputClass = "border border-ink-200 rounded-lg px-3 py-2 text-sm w-full";

export default function Team() {
  const [users, setUsers] = useState<HospitalUser[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("staff");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editEmail, setEditEmail] = useState("");
  const [editPassword, setEditPassword] = useState("");
  const [editError, setEditError] = useState<string | null>(null);

  function refresh() {
    api.listMyHospitalUsers({ limit: 100 }).then(setUsers);
  }

  useEffect(refresh, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim() || !password) return;
    setCreating(true);
    setError(null);
    try {
      await api.createMyHospitalUser({ email: email.trim(), password, role });
      setEmail("");
      setPassword("");
      setRole("staff");
      setShowAddForm(false);
      refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function changeRole(user: HospitalUser, newRole: string) {
    await api.updateMyHospitalUser(user.id, { role: newRole });
    refresh();
  }

  async function removeUser(user: HospitalUser) {
    if (!confirm(`Remove ${user.email}? This deletes their login.`)) return;
    await api.deleteMyHospitalUser(user.id);
    refresh();
  }

  function startEdit(user: HospitalUser) {
    setEditingId(user.id);
    setEditEmail(user.email);
    setEditPassword("");
    setEditError(null);
  }

  async function submitEdit(userId: string) {
    if (editPassword && editPassword.length < 8) {
      setEditError("Password must be at least 8 characters.");
      return;
    }
    try {
      await api.updateMyHospitalUser(userId, {
        email: editEmail.trim(),
        ...(editPassword ? { password: editPassword } : {}),
      });
      setEditingId(null);
      refresh();
    } catch (err) {
      setEditError((err as Error).message);
    }
  }

  return (
    <AppShell title="Team">
      <p className="text-sm text-ink-500 mb-5 max-w-2xl">
        Add and manage your hospital's staff and doctor logins. Adding another hospital admin
        isn't available here; contact the platform operator for that.
      </p>

      <Card className="p-6 max-w-3xl">
        <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
          <p className="font-medium text-ink-900">Users ({users.length})</p>
          <Button size="sm" onClick={() => setShowAddForm((v) => !v)}>
            {showAddForm ? <X size={16} /> : <Plus size={16} />}
            {showAddForm ? "Cancel" : "Add User"}
          </Button>
        </div>

        {showAddForm && (
          <form
            onSubmit={handleCreate}
            autoComplete="off"
            className="grid sm:grid-cols-[1fr_1fr_auto_auto] gap-2 items-start mb-5 bg-ink-50 border border-ink-200 rounded-lg p-3"
          >
            <input
              type="email"
              required
              name="new-team-member-email"
              autoComplete="off"
              placeholder="Email"
              className={inputClass}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <PasswordInput
              required
              name="new-team-member-password"
              autoComplete="new-password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <select className={inputClass} value={role} onChange={(e) => setRole(e.target.value)}>
              {ROLES.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </select>
            <Button type="submit" disabled={creating}>
              {creating ? "Creating…" : "Create"}
            </Button>
            {error && <p className="text-red-600 text-sm sm:col-span-4">{error}</p>}
          </form>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[420px]">
            <thead>
              <tr className="text-left text-ink-500 border-b border-ink-200">
                <th className="py-2 font-medium">Email</th>
                <th className="py-2 font-medium">Role</th>
                <th className="py-2 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <Fragment key={u.id}>
                  <tr className="border-b border-ink-100">
                    <td className="py-2.5 text-ink-900 break-all">{u.email}</td>
                    <td className="py-2.5">
                      <select
                        className="text-xs border border-ink-200 rounded-md px-1.5 py-1 capitalize"
                        value={u.role}
                        onChange={(e) => changeRole(u, e.target.value)}
                      >
                        {ROLES.map((r) => (
                          <option key={r.value} value={r.value}>
                            {r.label}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="py-2.5">
                      <div className="flex items-center justify-end gap-3">
                        <button
                          title="Edit email / password"
                          onClick={() => (editingId === u.id ? setEditingId(null) : startEdit(u))}
                          className="text-ink-400 hover:text-brand-600"
                        >
                          <Pencil size={16} />
                        </button>
                        <button
                          title="Remove user"
                          onClick={() => removeUser(u)}
                          className="text-ink-400 hover:text-red-600"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                  {editingId === u.id && (
                    <tr className="border-b border-ink-100 bg-ink-50">
                      <td colSpan={3} className="py-3 px-2">
                        <p className="text-xs text-ink-400 mb-1.5">
                          Passwords are encrypted and can never be viewed; leave blank to keep the
                          current one.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-2">
                          <input
                            type="email"
                            className={inputClass}
                            placeholder="Email"
                            value={editEmail}
                            onChange={(e) => setEditEmail(e.target.value)}
                          />
                          <PasswordInput
                            autoComplete="new-password"
                            placeholder="New password (min. 8 characters, optional)"
                            value={editPassword}
                            onChange={(e) => setEditPassword(e.target.value)}
                          />
                          <Button size="sm" onClick={() => submitEdit(u.id)}>
                            Save
                          </Button>
                        </div>
                        {editError && <p className="text-red-600 text-xs mt-1">{editError}</p>}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
              {users.length === 0 && (
                <tr>
                  <td colSpan={3} className="text-center text-ink-400 py-8">
                    No team members yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </AppShell>
  );
}
