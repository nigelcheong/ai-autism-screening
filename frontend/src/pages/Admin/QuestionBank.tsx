import { useEffect, useMemo, useState } from "react";
import { Plus, Pencil, Check, X, ChevronUp, ChevronDown, Trash2 } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { SearchInput } from "../../components/ui/SearchInput";
import { api, AdminQuestion } from "../../api/client";
import { INSTRUMENT_LABELS, Instrument } from "../../utils/age";

const INSTRUMENTS: Instrument[] = ["qchat10", "aq10"];

interface EditableOptions {
  options: string[];
  scoring: boolean[]; // same length as options; true = counts as concerning
}

function emptyOptions(): EditableOptions {
  return { options: ["", ""], scoring: [false, false] };
}

function OptionsEditor({
  value,
  onChange,
}: {
  value: EditableOptions;
  onChange: (v: EditableOptions) => void;
}) {
  function setOptionText(i: number, text: string) {
    const options = [...value.options];
    options[i] = text;
    onChange({ ...value, options });
  }
  function toggleScoring(i: number) {
    const scoring = [...value.scoring];
    scoring[i] = !scoring[i];
    onChange({ ...value, scoring });
  }
  function addOption() {
    onChange({ options: [...value.options, ""], scoring: [...value.scoring, false] });
  }
  function removeOption(i: number) {
    if (value.options.length <= 2) return;
    onChange({
      options: value.options.filter((_, idx) => idx !== i),
      scoring: value.scoring.filter((_, idx) => idx !== i),
    });
  }

  return (
    <div className="space-y-1.5">
      <span className="text-xs text-ink-500">
        Options, in order shown — check the ones that count toward risk:
      </span>
      {value.options.map((opt, i) => (
        <div key={i} className="flex items-center gap-2">
          <input
            type="checkbox"
            title="Counts toward risk"
            checked={value.scoring[i]}
            onChange={() => toggleScoring(i)}
          />
          <input
            className="flex-1 border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm"
            placeholder={`Option ${i + 1}`}
            value={opt}
            onChange={(e) => setOptionText(i, e.target.value)}
          />
          <button
            type="button"
            onClick={() => removeOption(i)}
            disabled={value.options.length <= 2}
            className="text-ink-300 hover:text-red-600 disabled:opacity-30"
          >
            <Trash2 size={14} />
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={addOption}
        className="text-xs text-brand-600 hover:underline"
      >
        + Add option
      </button>
    </div>
  );
}

export default function QuestionBank() {
  const [tab, setTab] = useState<Instrument>("qchat10");
  const [questions, setQuestions] = useState<AdminQuestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [editOptions, setEditOptions] = useState<EditableOptions>(emptyOptions());

  const [showAddForm, setShowAddForm] = useState(false);
  const [newText, setNewText] = useState("");
  const [newOptions, setNewOptions] = useState<EditableOptions>(emptyOptions());
  const [adding, setAdding] = useState(false);

  function refresh() {
    api
      .listQuestions()
      .then(setQuestions)
      .catch((err) => setError((err as Error).message));
  }

  useEffect(refresh, []);

  const instrumentQuestions = useMemo(
    () => questions.filter((q) => q.instrument === tab),
    [questions, tab]
  );

  const byKey = new Map<string, AdminQuestion[]>();
  for (const q of instrumentQuestions) {
    byKey.set(q.key, [...(byKey.get(q.key) ?? []), q]);
  }
  const keys = [...byKey.keys()].sort((a, b) => {
    const aOrder = Math.max(...(byKey.get(a) ?? []).map((q) => q.sort_order));
    const bOrder = Math.max(...(byKey.get(b) ?? []).map((q) => q.sort_order));
    return aOrder - bOrder;
  });

  function currentForKey(key: string): AdminQuestion {
    const versions = byKey.get(key) ?? [];
    return versions.find((v) => v.active) ?? versions[0];
  }

  const isSearching = search.trim().length > 0;
  const visibleKeys = isSearching
    ? keys.filter((k) => currentForKey(k).text.toLowerCase().includes(search.toLowerCase()))
    : keys;

  async function move(index: number, direction: -1 | 1) {
    const target = index + direction;
    if (target < 0 || target >= keys.length) return;
    const reordered = [...keys];
    [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
    await api.reorderQuestions(reordered);
    refresh();
  }

  function startEdit(q: AdminQuestion) {
    setEditingId(q.id);
    setEditText(q.text);
    setEditOptions({
      options: [...q.options],
      scoring: q.options.map((_, i) => q.scoring_indices.includes(i)),
    });
  }

  async function saveEdit(q: AdminQuestion) {
    const scoring_indices = editOptions.scoring
      .map((checked, i) => (checked ? i : -1))
      .filter((i) => i >= 0);
    await api.updateQuestion(q.id, {
      text: editText.trim(),
      options: editOptions.options.map((o) => o.trim()),
      scoring_indices,
    });
    setEditingId(null);
    refresh();
  }

  async function toggleActive(q: AdminQuestion) {
    await api.updateQuestion(q.id, { active: !q.active });
    refresh();
  }

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    if (!newText.trim() || newOptions.options.some((o) => !o.trim())) return;
    setAdding(true);
    try {
      const scoring_indices = newOptions.scoring
        .map((checked, i) => (checked ? i : -1))
        .filter((i) => i >= 0);
      await api.createQuestion({
        instrument: tab,
        text: newText.trim(),
        options: newOptions.options.map((o) => o.trim()),
        scoring_indices,
      });
      setNewText("");
      setNewOptions(emptyOptions());
      setShowAddForm(false);
      refresh();
    } finally {
      setAdding(false);
    }
  }

  return (
    <AppShell title="Question Bank">
      <p className="text-sm text-ink-500 mb-4 max-w-2xl">
        The two age-based screening instruments. Editing a question's wording or options creates
        a new version and retires the old one — past screenings keep showing exactly what the
        patient answered, in their report. Toggling a question inactive removes it from new
        screenings without deleting its history.
      </p>

      <div className="flex gap-2 mb-4">
        {INSTRUMENTS.map((inst) => (
          <button
            key={inst}
            onClick={() => setTab(inst)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
              tab === inst ? "bg-brand-600 text-white" : "bg-ink-100 text-ink-600 hover:bg-ink-200"
            }`}
          >
            {INSTRUMENT_LABELS[inst]}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 mb-4 max-w-2xl">
        <SearchInput
          className="flex-1 min-w-[180px]"
          placeholder="Search questions…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <Button onClick={() => setShowAddForm((v) => !v)}>
          {showAddForm ? <X size={16} /> : <Plus size={16} />}
          {showAddForm ? "Cancel" : "Add Question"}
        </Button>
      </div>

      {showAddForm && (
        <Card className="p-4 mb-5 max-w-2xl">
          <form onSubmit={handleAdd} className="space-y-3">
            <textarea
              required
              className="w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
              placeholder={`New ${INSTRUMENT_LABELS[tab]} question…`}
              rows={2}
              value={newText}
              onChange={(e) => setNewText(e.target.value)}
            />
            <OptionsEditor value={newOptions} onChange={setNewOptions} />
            <Button type="submit" disabled={adding}>
              {adding ? "Adding…" : "Add Question"}
            </Button>
          </form>
        </Card>
      )}

      {isSearching && (
        <p className="text-xs text-ink-400 mb-3">Reordering is disabled while searching.</p>
      )}

      {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

      <div className="space-y-3 max-w-2xl">
        {visibleKeys.map((key) => {
          const index = keys.indexOf(key);
          const versions = (byKey.get(key) ?? []).sort((a, b) => b.version - a.version);
          const current = versions.find((v) => v.active) ?? versions[0];
          const history = versions.filter((v) => v.id !== current.id);
          const isEditing = editingId === current.id;

          return (
            <Card key={key} className="p-4">
              <div className="flex items-start gap-3">
                <div className="flex flex-col items-center shrink-0 pt-0.5">
                  {isSearching ? (
                    <span className="text-ink-300 text-xs">{index + 1}</span>
                  ) : (
                    <>
                      <button
                        onClick={() => move(index, -1)}
                        disabled={index === 0}
                        className="text-ink-300 hover:text-ink-700 disabled:opacity-30"
                      >
                        <ChevronUp size={14} />
                      </button>
                      <span className="text-ink-400 text-xs">{index + 1}</span>
                      <button
                        onClick={() => move(index, 1)}
                        disabled={index === keys.length - 1}
                        className="text-ink-300 hover:text-ink-700 disabled:opacity-30"
                      >
                        <ChevronDown size={14} />
                      </button>
                    </>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  {isEditing ? (
                    <div className="space-y-2">
                      <textarea
                        className="w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        rows={2}
                      />
                      <OptionsEditor value={editOptions} onChange={setEditOptions} />
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => saveEdit(current)}>
                          <Check size={14} />
                          Save (new version)
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className={`text-sm ${current.active ? "text-ink-900" : "text-ink-400"}`}>
                        {current.text}
                      </p>
                      <ul className="mt-1.5 flex flex-wrap gap-1.5">
                        {current.options.map((opt, i) => (
                          <li
                            key={opt}
                            className={`text-xs px-2 py-0.5 rounded-full border ${
                              current.scoring_indices.includes(i)
                                ? "border-red-200 bg-red-50 text-red-700"
                                : "border-ink-200 text-ink-500"
                            }`}
                          >
                            {opt}
                          </li>
                        ))}
                      </ul>
                    </>
                  )}

                  {!isEditing && history.length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs text-ink-400 cursor-pointer hover:text-ink-600">
                        {history.length} earlier version{history.length > 1 ? "s" : ""}
                      </summary>
                      <ul className="mt-1 space-y-1">
                        {history.map((h) => (
                          <li key={h.id} className="text-xs text-ink-400 pl-3 border-l border-ink-200">
                            v{h.version}: {h.text}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>

                {!isEditing && (
                  <div className="flex items-center gap-2 shrink-0">
                    <Badge tone={current.active ? "low" : "neutral"}>
                      v{current.version} · {current.active ? "active" : "retired"}
                    </Badge>
                    <button
                      title="Edit"
                      onClick={() => startEdit(current)}
                      className="text-ink-400 hover:text-brand-600"
                    >
                      <Pencil size={16} />
                    </button>
                    <button
                      onClick={() => toggleActive(current)}
                      className="text-xs text-brand-600 hover:underline whitespace-nowrap"
                    >
                      {current.active ? "Deactivate" : "Activate"}
                    </button>
                  </div>
                )}
              </div>
            </Card>
          );
        })}
        {visibleKeys.length === 0 && !error && (
          <p className="text-center text-ink-400 text-sm py-8">
            {keys.length === 0 ? "No questions yet." : "No questions match your search."}
          </p>
        )}
      </div>
    </AppShell>
  );
}
