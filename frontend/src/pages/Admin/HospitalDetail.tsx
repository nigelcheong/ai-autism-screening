import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronRight, KeyRound, Copy, Check, Building2, UserCircle, Pencil } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { api, Hospital } from "../../api/client";
import { HospitalUsersPanel } from "./HospitalUsersPanel";

const inputClass = "mt-1 w-full border border-ink-200 rounded-lg px-3 py-2 text-sm";

export default function HospitalDetail() {
  const { hospitalId } = useParams<{ hospitalId: string }>();
  const [hospital, setHospital] = useState<Hospital | null>(null);
  const [keyStatus, setKeyStatus] = useState<{ active: boolean; masked_key: string | null } | null>(
    null
  );
  const [newKey, setNewKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [editingName, setEditingName] = useState(false);
  const [nameForm, setNameForm] = useState("");
  const [savingName, setSavingName] = useState(false);
  const [nameError, setNameError] = useState<string | null>(null);

  const [editingContact, setEditingContact] = useState(false);
  const [contactForm, setContactForm] = useState({
    contact_name: "",
    contact_designation: "",
    contact_email: "",
    contact_phone: "",
  });
  const [savingContact, setSavingContact] = useState(false);
  const [contactError, setContactError] = useState<string | null>(null);

  function refresh() {
    if (!hospitalId) return;
    api
      .getHospital(hospitalId)
      .then((h) => {
        setHospital(h);
        setNameForm(h.name);
        setContactForm({
          contact_name: h.contact_name ?? "",
          contact_designation: h.contact_designation ?? "",
          contact_email: h.contact_email ?? "",
          contact_phone: h.contact_phone ?? "",
        });
      })
      .catch((err) => setError((err as Error).message));
    api.getHospitalApiKeyStatusAsAdmin(hospitalId).then(setKeyStatus);
  }

  useEffect(refresh, [hospitalId]);

  async function toggleActive() {
    if (!hospitalId || !hospital) return;
    const updated = await api.updateHospital(hospitalId, { active: !hospital.active });
    setHospital(updated);
  }

  async function saveName(e: React.FormEvent) {
    e.preventDefault();
    if (!hospitalId || !nameForm.trim()) return;
    setSavingName(true);
    setNameError(null);
    try {
      const updated = await api.updateHospital(hospitalId, { name: nameForm.trim() });
      setHospital(updated);
      setEditingName(false);
    } catch (err) {
      setNameError((err as Error).message);
    } finally {
      setSavingName(false);
    }
  }

  async function saveContact(e: React.FormEvent) {
    e.preventDefault();
    if (!hospitalId) return;
    setSavingContact(true);
    setContactError(null);
    try {
      const updated = await api.updateHospital(hospitalId, contactForm);
      setHospital(updated);
      setEditingContact(false);
    } catch (err) {
      setContactError((err as Error).message);
    } finally {
      setSavingContact(false);
    }
  }

  async function issueKey() {
    if (!hospitalId) return;
    const result = await api.issueHospitalApiKeyAsAdmin(hospitalId);
    setNewKey(result.api_key);
    const status = await api.getHospitalApiKeyStatusAsAdmin(hospitalId);
    setKeyStatus(status);
  }

  function copyKey() {
    if (!newKey) return;
    navigator.clipboard.writeText(newKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  const hasContact =
    hospital?.contact_name || hospital?.contact_email || hospital?.contact_phone;

  return (
    <AppShell title="Manage Hospital">
      <div className="flex items-center gap-1.5 text-sm text-ink-500 mb-5">
        <Link to="/platform/hospitals" className="hover:text-ink-900">
          Hospitals
        </Link>
        <ChevronRight size={14} />
        <span className="text-ink-900 font-medium">{hospital?.name ?? "…"}</span>
      </div>

      {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

      {hospital && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card className="p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
                  <Building2 size={20} />
                </div>
                <div className="min-w-0 flex-1">
                  {editingName ? (
                    <form onSubmit={saveName} className="space-y-2">
                      <input
                        autoFocus
                        className={inputClass}
                        value={nameForm}
                        onChange={(e) => setNameForm(e.target.value)}
                      />
                      {nameError && <p className="text-red-600 text-xs">{nameError}</p>}
                      <div className="flex gap-2">
                        <Button type="submit" size="sm" disabled={savingName}>
                          {savingName ? "Saving…" : "Save"}
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setEditingName(false);
                            setNameForm(hospital.name);
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                    </form>
                  ) : (
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-ink-900 truncate">{hospital.name}</p>
                      <button
                        onClick={() => setEditingName(true)}
                        className="text-ink-400 hover:text-brand-600 shrink-0"
                        title="Edit hospital name"
                      >
                        <Pencil size={14} />
                      </button>
                    </div>
                  )}
                  {!editingName && (
                    <Badge tone={hospital.active ? "low" : "neutral"}>
                      {hospital.active ? "Active" : "Inactive"}
                    </Badge>
                  )}
                </div>
              </div>

              <Button variant="secondary" onClick={toggleActive} className="w-full">
                {hospital.active ? "Deactivate Hospital" : "Activate Hospital"}
              </Button>
            </Card>

            <Card className="p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
                    <UserCircle size={20} />
                  </div>
                  <p className="font-medium text-ink-900">Contact Person</p>
                </div>
                {!editingContact && (
                  <button
                    onClick={() => setEditingContact(true)}
                    className="text-ink-400 hover:text-brand-600"
                    title="Edit contact"
                  >
                    <Pencil size={16} />
                  </button>
                )}
              </div>

              {editingContact ? (
                <form onSubmit={saveContact} className="space-y-2">
                  <input
                    className={inputClass}
                    placeholder="Name"
                    value={contactForm.contact_name}
                    onChange={(e) =>
                      setContactForm((f) => ({ ...f, contact_name: e.target.value }))
                    }
                  />
                  <input
                    className={inputClass}
                    placeholder="Designation"
                    value={contactForm.contact_designation}
                    onChange={(e) =>
                      setContactForm((f) => ({ ...f, contact_designation: e.target.value }))
                    }
                  />
                  <input
                    type="email"
                    className={inputClass}
                    placeholder="Email"
                    value={contactForm.contact_email}
                    onChange={(e) =>
                      setContactForm((f) => ({ ...f, contact_email: e.target.value }))
                    }
                  />
                  <input
                    type="tel"
                    className={inputClass}
                    placeholder="Phone number"
                    value={contactForm.contact_phone}
                    onChange={(e) =>
                      setContactForm((f) => ({ ...f, contact_phone: e.target.value }))
                    }
                  />
                  {contactError && <p className="text-red-600 text-xs">{contactError}</p>}
                  <div className="flex gap-2 pt-1">
                    <Button type="submit" size="sm" disabled={savingContact}>
                      {savingContact ? "Saving…" : "Save"}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      onClick={() => setEditingContact(false)}
                    >
                      Cancel
                    </Button>
                  </div>
                </form>
              ) : hasContact ? (
                <div className="text-sm space-y-1">
                  <p className="text-ink-900 font-medium">{hospital.contact_name || "—"}</p>
                  {hospital.contact_designation && (
                    <p className="text-ink-500">{hospital.contact_designation}</p>
                  )}
                  {hospital.contact_email && (
                    <p className="text-ink-500">{hospital.contact_email}</p>
                  )}
                  {hospital.contact_phone && (
                    <p className="text-ink-500">{hospital.contact_phone}</p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-ink-400">No contact person on file.</p>
              )}
            </Card>

            <Card className="p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
                  <KeyRound size={20} />
                </div>
                <div>
                  <p className="font-medium text-ink-900">API Key</p>
                  {keyStatus && (
                    <Badge tone={keyStatus.active ? "low" : "neutral"}>
                      {keyStatus.active ? "Active" : "No key issued"}
                    </Badge>
                  )}
                </div>
              </div>

              {keyStatus?.masked_key && (
                <div className="bg-ink-50 border border-ink-200 rounded-lg px-4 py-3 font-mono text-sm text-ink-700 mb-3">
                  {keyStatus.masked_key}
                </div>
              )}

              {newKey && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 mb-3">
                  <p className="text-xs font-medium text-amber-800 mb-1">
                    New full key — copy it now, it won't be shown again:
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

              <Button variant="secondary" onClick={issueKey} className="w-full">
                {keyStatus?.active ? "Regenerate Key" : "Issue Key"}
              </Button>
            </Card>
          </div>

          <HospitalUsersPanel hospitalId={hospital.id} />
        </div>
      )}
    </AppShell>
  );
}
