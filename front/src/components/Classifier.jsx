import { useState, useRef, useEffect } from "react";
import { gsap } from "gsap";
import { predict } from "../api";
import Explanation from "./Explanation";
import {
  WORKCLASS, EDUCATION, MARITAL_STATUS, OCCUPATION,
  RELATIONSHIP, RACE, NATIVE_COUNTRY,
} from "../constants";

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

const MODEL_COLORS = {
  logistic_regression: "#6366f1",
  decision_tree:       "#f59e0b",
  knn:                 "#10b981",
};

function Field({ label, hint, children }) {
  return (
    <div className="field">
      <label>{label}{hint && <span className="field-hint">{hint}</span>}</label>
      {children}
    </div>
  );
}

function ProbGauge({ value, color }) {
  const pct = Math.round(value * 100);
  const barRef = useRef(null);
  useEffect(() => {
    if (!barRef.current) return;
    gsap.fromTo(barRef.current, { width: "0%" }, { width: `${pct}%`, duration: 0.8, ease: "power2.out" });
  }, [pct]);
  return (
    <div className="gauge-wrap">
      <div className="gauge-track">
        <div ref={barRef} className="gauge-fill" style={{ background: color }} />
      </div>
      <span className="gauge-pct" style={{ color }}>{pct}%</span>
    </div>
  );
}

function ResultCard({ result, index }) {
  const cardRef = useRef(null);
  const isHigh  = result.prediction === 1;
  const color   = MODEL_COLORS[result.model_id] ?? "#64748b";

  useEffect(() => {
    gsap.fromTo(cardRef.current,
      { y: 40, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.5, delay: index * 0.12, ease: "power3.out" }
    );
  }, []);

  return (
    <div ref={cardRef} className={`result-card ${isHigh ? "high" : "low"}`} style={{ "--accent": color }}>
      <div className="rc-header">
        <div className="rc-dot" style={{ background: color }} />
        <span className="rc-model">{result.model_name}</span>
      </div>
      <div className="rc-verdict" style={{ color: isHigh ? "#10b981" : "#94a3b8" }}>
        {result.label}
      </div>
      <div className="rc-probs">
        <div className="rc-prob-row">
          <span className="rc-prob-label">≤ 50K</span>
          <ProbGauge value={result.probability_le50k} color="#94a3b8" />
        </div>
        <div className="rc-prob-row">
          <span className="rc-prob-label">&gt; 50K</span>
          <ProbGauge value={result.probability_gt50k} color={isHigh ? "#10b981" : "#cbd5e1"} />
        </div>
      </div>
      <div className="rc-conf">
        Confianza: <strong style={{ color }}>{Math.round(Math.max(result.probability_gt50k, result.probability_le50k) * 100)}%</strong>
      </div>
    </div>
  );
}

function Consensus({ results }) {
  const votes = results.filter(r => r.prediction === 1).length;
  const total = results.length;
  const verdict = votes > total / 2 ? ">50K" : "≤50K";
  const confidence = Math.round(
    results.reduce((s, r) => s + (r.prediction === 1 ? r.probability_gt50k : r.probability_le50k), 0) / total * 100
  );
  const ref = useRef(null);
  useEffect(() => {
    gsap.fromTo(ref.current, { scale: 0.8, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.5, ease: "back.out(1.5)" });
  }, []);
  return (
    <div ref={ref} className={`consensus ${verdict === ">50K" ? "high" : "low"}`}>
      <div className="consensus-label">Consenso ({votes}/{total} modelos)</div>
      <div className="consensus-verdict">{verdict}</div>
      <div className="consensus-conf">Confianza promedio: {confidence}%</div>
    </div>
  );
}

export default function Classifier({ modelsData }) {
  const [form, setForm]               = useState(DEFAULT_FORM);
  const [selectedModel, setSelected]  = useState("all");
  const [results, setResults]         = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const formRef = useRef(null);

  useEffect(() => {
    gsap.fromTo(
      formRef.current.querySelectorAll("section"),
      { x: -20, opacity: 0 },
      { x: 0, opacity: 1, duration: 0.5, stagger: 0.08, ease: "power2.out" }
    );
  }, []);

  const handleChange = e => {
    const { name, value } = e.target;
    const nums = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"];
    setForm(f => ({ ...f, [name]: nums.includes(name) ? Number(value) : value }));
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true); setError(null); setResults(null); setExplanation(null);
    try {
      const model = selectedModel === "all" ? null : selectedModel;
      const data  = await predict(form, model);
      setResults(data.results);
      setExplanation(data.explanation);
    } catch (err) {
      setError(err?.response?.data?.detail || "No se pudo conectar con el backend. ¿Está corriendo en el puerto 8000?");
    } finally {
      setLoading(false);
    }
  };

  const sel = (label, name, val, opts) => (
    <Field label={label}>
      <select name={name} value={val} onChange={handleChange}>
        {opts.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </Field>
  );
  const num = (label, name, val, min, max, hint) => (
    <Field label={label} hint={hint}>
      <input type="number" name={name} value={val} onChange={handleChange} min={min} max={max} />
    </Field>
  );

  return (
    <div className="classifier-layout">

      <form ref={formRef} className="form-panel" onSubmit={handleSubmit}>
        <section>
          <h3 className="section-title">Datos personales</h3>
          <div className="fields-grid">
            {num("Edad", "age", form.age, 17, 90)}
            {sel("Sexo", "sex", form.sex, ["Male", "Female"])}
            {sel("Raza", "race", form.race, RACE)}
            {sel("País de origen", "native-country", form["native-country"], NATIVE_COUNTRY)}
          </div>
        </section>

        <section>
          <h3 className="section-title">Educación y trabajo</h3>
          <div className="fields-grid">
            {sel("Nivel educativo", "education", form.education, EDUCATION)}
            {num("Años educación", "education-num", form["education-num"], 1, 16, "1-16")}
            {sel("Tipo de trabajo", "workclass", form.workclass, WORKCLASS)}
            {sel("Ocupación", "occupation", form.occupation, OCCUPATION)}
            {num("Horas / semana", "hours-per-week", form["hours-per-week"], 1, 99)}
          </div>
        </section>

        <section>
          <h3 className="section-title">Familia</h3>
          <div className="fields-grid">
            {sel("Estado civil", "marital-status", form["marital-status"], MARITAL_STATUS)}
            {sel("Relación familiar", "relationship", form.relationship, RELATIONSHIP)}
          </div>
        </section>

        <section>
          <h3 className="section-title">Capital</h3>
          <div className="fields-grid">
            {num("Ganancia de capital ($)", "capital-gain", form["capital-gain"], 0, 99999)}
            {num("Pérdida de capital ($)", "capital-loss", form["capital-loss"], 0, 99999)}
          </div>
        </section>

        <section>
          <h3 className="section-title">Modelo</h3>
          <div className="model-chips">
            {[
              { id: "all",                 label: "Todos",              color: "#64748b" },
              { id: "logistic_regression", label: "Reg. Logística",     color: MODEL_COLORS.logistic_regression },
              { id: "decision_tree",       label: "Árbol de Decisión",  color: MODEL_COLORS.decision_tree },
              { id: "knn",                 label: "KNN",                color: MODEL_COLORS.knn },
            ].map(({ id, label, color }) => (
              <label key={id} className={`chip ${selectedModel === id ? "chip-active" : ""}`}
                     style={selectedModel === id ? { background: color, borderColor: color } : {}}>
                <input type="radio" name="model" value={id} checked={selectedModel === id}
                       onChange={() => setSelected(id)} />
                {label}
              </label>
            ))}
          </div>
        </section>

        <button className="btn-predict" type="submit" disabled={loading}>
          {loading ? <span className="btn-spinner" /> : "Clasificar"}
        </button>
      </form>

      <div className="results-panel">
        {error && <div className="error-box">{error}</div>}

        {!results && !error && (
          <div className="empty-state">
            <div className="empty-icon">◈</div>
            <p>Completa los datos y presiona <strong>Clasificar</strong></p>
            {modelsData && (
              <div className="quick-metrics">
                {modelsData.metrics.map(m => (
                  <div key={m.id} className="qm-card" style={{ "--c": MODEL_COLORS[m.id] }}>
                    <div className="qm-name">{m.name}</div>
                    <div className="qm-auc">AUC {m.auc}</div>
                    <div className="qm-acc">Acc {(m.accuracy * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {results && (
          <>
            {results.length > 1 && <Consensus results={results} />}
            <div className="results-grid">
              {results.map((r, i) => <ResultCard key={r.model_id} result={r} index={i} />)}
            </div>
            {explanation && <Explanation explanation={explanation} />}
          </>
        )}
      </div>
    </div>
  );
}
