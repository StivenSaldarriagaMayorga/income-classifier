import { useState, useEffect } from "react";
import { getModels, predict } from "./api";
import {
  WORKCLASS, EDUCATION, MARITAL_STATUS, OCCUPATION,
  RELATIONSHIP, RACE, NATIVE_COUNTRY,
} from "./constants";
import "./App.css";

const DEFAULT_FORM = {
  age: 35,
  workclass: "Private",
  education: "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  occupation: "Tech-support",
  relationship: "Not-in-family",
  race: "White",
  sex: "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States",
};

function Field({ label, children }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
    </div>
  );
}

function SelectInput({ label, name, value, onChange, options }) {
  return (
    <Field label={label}>
      <select name={name} value={value} onChange={onChange}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </Field>
  );
}

function NumberInput({ label, name, value, onChange, min, max }) {
  return (
    <Field label={label}>
      <input type="number" name={name} value={value} onChange={onChange} min={min} max={max} />
    </Field>
  );
}

function ProbBar({ value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar" style={{ width: `${pct}%`, background: color }} />
      <span className="prob-label">{pct}%</span>
    </div>
  );
}

function ResultCard({ result }) {
  const isHigh = result.prediction === 1;
  return (
    <div className={`result-card ${isHigh ? "high" : "low"}`}>
      <div className="result-model">{result.model_name}</div>
      <div className="result-label">{result.label}</div>
      <div className="prob-row">
        <span className="prob-title">≤50K</span>
        <ProbBar value={result.probability_le50k} color="#64748b" />
      </div>
      <div className="prob-row">
        <span className="prob-title">&gt;50K</span>
        <ProbBar value={result.probability_gt50k} color={isHigh ? "#16a34a" : "#94a3b8"} />
      </div>
    </div>
  );
}

function MetricsTable({ metrics }) {
  if (!metrics) return null;
  return (
    <div className="metrics-wrap">
      <h3>Métricas en test (entrenamiento previo)</h3>
      <table className="metrics-table">
        <thead>
          <tr><th>Modelo</th><th>AUC-ROC</th><th>Accuracy</th><th>F1</th></tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m.name}>
              <td>{m.name}</td>
              <td>{m.auc}</td>
              <td>{m.accuracy}</td>
              <td>{m.f1}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [selectedModel, setSelectedModel] = useState("all");
  const [results, setResults] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModels().then((data) => setMetrics(data.metrics)).catch(() => {});
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    const numFields = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"];
    setForm((f) => ({ ...f, [name]: numFields.includes(name) ? Number(value) : value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const model = selectedModel === "all" ? null : selectedModel;
      const data = await predict(form, model);
      setResults(data.results);
    } catch (err) {
      setError(err?.response?.data?.detail || "Error al conectar con el backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Clasificador de Ingresos</h1>
        <p>Predice si el ingreso anual supera los $50K usando modelos de Machine Learning</p>
      </header>

      <div className="layout">
        <form className="form-panel" onSubmit={handleSubmit}>
          <section className="form-section">
            <h2>Datos personales</h2>
            <NumberInput label="Edad" name="age" value={form.age} onChange={handleChange} min={17} max={90} />
            <SelectInput label="Sexo" name="sex" value={form.sex} onChange={handleChange} options={["Male", "Female"]} />
            <SelectInput label="Raza" name="race" value={form.race} onChange={handleChange} options={RACE} />
            <SelectInput label="País de origen" name="native-country" value={form["native-country"]} onChange={handleChange} options={NATIVE_COUNTRY} />
          </section>

          <section className="form-section">
            <h2>Educación y trabajo</h2>
            <SelectInput label="Nivel educativo" name="education" value={form.education} onChange={handleChange} options={EDUCATION} />
            <NumberInput label="Años de educación (1-16)" name="education-num" value={form["education-num"]} onChange={handleChange} min={1} max={16} />
            <SelectInput label="Tipo de trabajo" name="workclass" value={form.workclass} onChange={handleChange} options={WORKCLASS} />
            <SelectInput label="Ocupación" name="occupation" value={form.occupation} onChange={handleChange} options={OCCUPATION} />
            <NumberInput label="Horas/semana" name="hours-per-week" value={form["hours-per-week"]} onChange={handleChange} min={1} max={99} />
          </section>

          <section className="form-section">
            <h2>Estado civil</h2>
            <SelectInput label="Estado civil" name="marital-status" value={form["marital-status"]} onChange={handleChange} options={MARITAL_STATUS} />
            <SelectInput label="Relación familiar" name="relationship" value={form.relationship} onChange={handleChange} options={RELATIONSHIP} />
          </section>

          <section className="form-section">
            <h2>Capital</h2>
            <NumberInput label="Ganancia de capital ($)" name="capital-gain" value={form["capital-gain"]} onChange={handleChange} min={0} max={99999} />
            <NumberInput label="Pérdida de capital ($)" name="capital-loss" value={form["capital-loss"]} onChange={handleChange} min={0} max={99999} />
          </section>

          <section className="form-section">
            <h2>Seleccionar modelo</h2>
            <div className="model-selector">
              {[
                { id: "all", label: "Todos los modelos" },
                { id: "logistic_regression", label: "Regresión Logística" },
                { id: "decision_tree", label: "Árbol de Decisión" },
                { id: "knn", label: "KNN" },
              ].map(({ id, label }) => (
                <label key={id} className={`model-chip ${selectedModel === id ? "active" : ""}`}>
                  <input type="radio" name="model" value={id} checked={selectedModel === id} onChange={() => setSelectedModel(id)} />
                  {label}
                </label>
              ))}
            </div>
          </section>

          <button className="btn-predict" type="submit" disabled={loading}>
            {loading ? "Clasificando…" : "Clasificar"}
          </button>
        </form>

        <div className="results-panel">
          {error && <div className="error-box">{error}</div>}

          {results && (
            <div>
              <h2>Resultados</h2>
              <div className="results-grid">
                {results.map((r) => <ResultCard key={r.model_id} result={r} />)}
              </div>
            </div>
          )}

          {!results && !error && (
            <div className="empty-state">
              <div className="empty-icon">🧮</div>
              <p>Completa el formulario y presiona <strong>Clasificar</strong></p>
            </div>
          )}

          <MetricsTable metrics={metrics} />
        </div>
      </div>
    </div>
  );
}
