import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

gsap.registerPlugin(ScrollTrigger);

const MODEL_COLORS = {
  logistic_regression: "#6366f1",
  decision_tree:       "#f59e0b",
  knn:                 "#10b981",
};

function StatCard({ label, value, sub, color, index }) {
  const ref     = useRef(null);
  const numRef  = useRef(null);
  const isFloat = String(value).includes(".");

  useEffect(() => {
    gsap.fromTo(ref.current,
      { y: 30, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.5, delay: index * 0.1, ease: "power3.out",
        scrollTrigger: { trigger: ref.current, start: "top 90%" } }
    );
    const target = isFloat ? parseFloat(value) : parseInt(value);
    gsap.fromTo({ val: 0 }, { val: target },
      { val: target, duration: 1.2, delay: index * 0.1 + 0.2, ease: "power2.out",
        onUpdate() {
          if (!numRef.current) return;
          numRef.current.textContent = isFloat
            ? this.targets()[0].val.toFixed(4)
            : Math.round(this.targets()[0].val).toLocaleString();
        }
      }
    );
  }, []);

  return (
    <div ref={ref} className="stat-card" style={{ "--border-color": color }}>
      <div className="stat-label">{label}</div>
      <div ref={numRef} className="stat-value" style={{ color }}>0</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function RocChart({ rocData }) {
  const points = [];
  rocData.forEach(model => {
    model.fpr.forEach((fpr, i) => {
      const existing = points.find(p => Math.abs(p.fpr - fpr) < 0.005);
      if (existing) {
        existing[model.id] = model.tpr[i];
      } else {
        points.push({ fpr, [model.id]: model.tpr[i] });
      }
    });
  });
  points.sort((a, b) => a.fpr - b.fpr);

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={points} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="fpr" tickFormatter={v => v.toFixed(1)} label={{ value: "FPR", position: "insideBottom", offset: -4, fill: "#64748b" }} tick={{ fill: "#64748b", fontSize: 11 }} />
        <YAxis tickFormatter={v => v.toFixed(1)} label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#64748b" }} tick={{ fill: "#64748b", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
          labelFormatter={v => `FPR: ${Number(v).toFixed(3)}`}
          formatter={(v, name) => [Number(v).toFixed(3), rocData.find(m => m.id === name)?.name ?? name]}
        />
        <Legend wrapperStyle={{ paddingTop: 12, fontSize: 13 }} formatter={name => rocData.find(m => m.id === name)?.name ?? name} />
        <ReferenceLine x={0} y={0} stroke="#334155" strokeDasharray="4 4" ifOverflow="extendDomain" />
        {rocData.map(m => (
          <Line key={m.id} type="monotone" dataKey={m.id} stroke={MODEL_COLORS[m.id]}
                strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
        ))}

        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#334155" strokeDasharray="4 4" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ConfusionMatrix({ data }) {
  const max = Math.max(data.tp, data.tn, data.fp, data.fn);
  const bg  = v => `rgba(99,102,241,${0.1 + 0.7 * (v / max)})`;
  const cells = [
    { label: "TN", value: data.tn, desc: "≤50K → ≤50K", good: true  },
    { label: "FP", value: data.fp, desc: "≤50K → >50K", good: false },
    { label: "FN", value: data.fn, desc: ">50K → ≤50K", good: false },
    { label: "TP", value: data.tp, desc: ">50K → >50K", good: true  },
  ];
  return (
    <div className="cm-grid">
      {cells.map(c => (
        <div key={c.label} className={`cm-cell ${c.good ? "cm-good" : "cm-bad"}`}
             style={{ background: c.good ? bg(c.value) : `rgba(239,68,68,${0.1 + 0.5 * (c.value / max)})` }}>
          <div className="cm-cell-label">{c.label}</div>
          <div className="cm-cell-value">{c.value.toLocaleString()}</div>
          <div className="cm-cell-desc">{c.desc}</div>
        </div>
      ))}
    </div>
  );
}

function MetricsRadar({ metrics }) {
  const data = ["accuracy", "f1", "auc", "precision", "recall"].map(key => ({
    metric: key.charAt(0).toUpperCase() + key.slice(1),
    ...Object.fromEntries(metrics.map(m => [m.id, m[key]])),
  }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data}>
        <PolarGrid stroke="#1e293b" />
        <PolarAngleAxis dataKey="metric" tick={{ fill: "#64748b", fontSize: 12 }} />
        {metrics.map(m => (
          <Radar key={m.id} name={m.name} dataKey={m.id}
                 stroke={MODEL_COLORS[m.id]} fill={MODEL_COLORS[m.id]} fillOpacity={0.15} strokeWidth={2} />
        ))}
        <Legend wrapperStyle={{ fontSize: 13 }} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                 formatter={v => v.toFixed(4)} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function FeatureImportance({ data }) {
  const top = data.slice(0, 12);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={top} layout="vertical" margin={{ left: 130, right: 20, top: 5, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
        <XAxis type="number" tick={{ fill: "#64748b", fontSize: 11 }} />
        <YAxis type="category" dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 11 }} width={125} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                 formatter={v => [v.toFixed(4), "Importancia"]} />
        <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function Dashboard({ modelsData, analytics }) {
  const [activeModel, setActiveModel] = useState(0);
  const sectionsRef = useRef([]);

  useEffect(() => {
    sectionsRef.current.forEach((el, i) => {
      if (!el) return;
      gsap.fromTo(el,
        { y: 40, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: "power3.out",
          scrollTrigger: { trigger: el, start: "top 88%", once: true } }
      );
    });
  }, []);

  if (!modelsData || !analytics) return <div className="loading-screen"><div className="spinner" /></div>;

  const { metrics, dataset } = modelsData;
  const { roc_curves, confusion_matrices, feature_importance } = analytics;
  const cm = confusion_matrices[activeModel];

  return (
    <div className="dashboard">

      <section ref={el => sectionsRef.current[0] = el} className="dash-section">
        <h2 className="dash-title">Resumen del dataset</h2>
        <div className="stats-grid">
          <StatCard label="Total registros" value={dataset.total_samples} color="#6366f1" index={0} />
          <StatCard label="Entrenamiento"   value={dataset.train_samples} color="#10b981" index={1} />
          <StatCard label="Test"            value={dataset.test_samples}  color="#f59e0b" index={2} />
          <StatCard label="Features"        value={dataset.n_features}    color="#ec4899" index={3} />
          <StatCard label="≤50K (clase 0)"  value={dataset.class_balance.le50k} sub="Clase mayoritaria" color="#94a3b8" index={4} />
          <StatCard label=">50K (clase 1)"  value={dataset.class_balance.gt50k} sub="Clase minoritaria" color="#f97316" index={5} />
        </div>
      </section>

      <section ref={el => sectionsRef.current[1] = el} className="dash-section">
        <h2 className="dash-title">Comparativa de modelos</h2>
        <div className="metrics-table-wrap">
          <table className="metrics-table">
            <thead>
              <tr><th>Modelo</th><th>AUC-ROC</th><th>Accuracy</th><th>F1</th><th>Precisión</th><th>Recall</th></tr>
            </thead>
            <tbody>
              {metrics.map(m => (
                <tr key={m.id}>
                  <td><span className="model-dot" style={{ background: MODEL_COLORS[m.id] }} />{m.name}</td>
                  <td><MetricBadge value={m.auc} best={Math.max(...metrics.map(x => x.auc))} /></td>
                  <td><MetricBadge value={m.accuracy} best={Math.max(...metrics.map(x => x.accuracy))} /></td>
                  <td><MetricBadge value={m.f1} best={Math.max(...metrics.map(x => x.f1))} /></td>
                  <td><MetricBadge value={m.precision} best={Math.max(...metrics.map(x => x.precision))} /></td>
                  <td><MetricBadge value={m.recall} best={Math.max(...metrics.map(x => x.recall))} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section ref={el => sectionsRef.current[2] = el} className="dash-section">
        <h2 className="dash-title">Radar de métricas</h2>
        <div className="chart-card">
          <MetricsRadar metrics={metrics} />
        </div>
      </section>

      <section ref={el => sectionsRef.current[3] = el} className="dash-section">
        <h2 className="dash-title">Curvas ROC</h2>
        <div className="chart-card">
          <div className="roc-aucs">
            {roc_curves.map(m => (
              <div key={m.id} className="roc-auc-badge" style={{ borderColor: MODEL_COLORS[m.id] }}>
                <span style={{ color: MODEL_COLORS[m.id] }}>{m.name}</span>
                <strong>AUC = {m.auc}</strong>
              </div>
            ))}
          </div>
          <RocChart rocData={roc_curves} />
          <p className="chart-note">La línea punteada representa un clasificador aleatorio (AUC = 0.5). Cuanto más arriba y a la izquierda, mejor.</p>
        </div>
      </section>

      <section ref={el => sectionsRef.current[4] = el} className="dash-section">
        <h2 className="dash-title">Matriz de confusión</h2>
        <div className="chart-card">
          <div className="cm-tabs">
            {confusion_matrices.map((m, i) => (
              <button key={m.id} className={`cm-tab ${activeModel === i ? "cm-tab-active" : ""}`}
                      style={activeModel === i ? { borderColor: MODEL_COLORS[m.id], color: MODEL_COLORS[m.id] } : {}}
                      onClick={() => setActiveModel(i)}>
                {m.name}
              </button>
            ))}
          </div>
          <ConfusionMatrix data={cm} />
          <div className="cm-legend">
            <span><strong>TN</strong> — Verdadero Negativo (≤50K correctamente clasificado)</span>
            <span><strong>FP</strong> — Falso Positivo (Error tipo I)</span>
            <span><strong>FN</strong> — Falso Negativo (Error tipo II)</span>
            <span><strong>TP</strong> — Verdadero Positivo (&gt;50K correctamente clasificado)</span>
          </div>
        </div>
      </section>

      <section ref={el => sectionsRef.current[5] = el} className="dash-section">
        <h2 className="dash-title">Importancia de variables <span className="title-sub">(coeficientes |β| — Reg. Logística)</span></h2>
        <div className="chart-card">
          <FeatureImportance data={feature_importance} />
        </div>
      </section>
    </div>
  );
}

function MetricBadge({ value, best }) {
  const isTop = value === best;
  return (
    <span className={`metric-badge ${isTop ? "metric-top" : ""}`}>{value}</span>
  );
}
