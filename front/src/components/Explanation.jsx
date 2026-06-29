import { useEffect, useRef } from "react";
import { gsap } from "gsap";

function FactorBar({ factor, max, positive, delay }) {
  const barRef = useRef(null);
  const pct = Math.min(100, Math.round((Math.abs(factor.contribution) / max) * 100));
  const color = positive ? "#10b981" : "#f87171";

  useEffect(() => {
    if (!barRef.current) return;
    gsap.fromTo(barRef.current, { width: "0%" },
      { width: `${pct}%`, duration: 0.7, delay, ease: "power2.out" });
  }, [pct, delay]);

  return (
    <div className="factor-row">
      <div className="factor-head">
        <span className="factor-label">
          {factor.label}
          {factor.detail && <span className="factor-detail"> · {factor.detail}</span>}
        </span>
        <span className="factor-impact" style={{ color }}>
          {positive ? "+" : "−"}{Math.abs(factor.contribution).toFixed(2)}
        </span>
      </div>
      <div className="factor-track">
        <div ref={barRef} className="factor-fill" style={{ background: color }} />
      </div>
    </div>
  );
}

export default function Explanation({ explanation }) {
  const ref = useRef(null);
  const allContribs = [...explanation.positive_factors, ...explanation.negative_factors]
    .map(f => Math.abs(f.contribution));
  const max = Math.max(...allContribs, 0.01);

  useEffect(() => {
    gsap.fromTo(ref.current, { y: 30, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.6, ease: "power3.out" });
  }, []);

  const confColor = {
    alta: "#10b981", moderada: "#f59e0b", baja: "#f87171",
  }[explanation.confidence_level] ?? "#64748b";

  return (
    <div ref={ref} className="explanation-card">
      <div className="exp-header">
        <div className="exp-icon">🧠</div>
        <div>
          <div className="exp-title">¿Cómo se tomó esta decisión?</div>
          <div className="exp-subtitle">Análisis explicable · modelo lineal auditable</div>
        </div>
        <div className="exp-conf" style={{ borderColor: confColor, color: confColor }}>
          Confianza {explanation.confidence_level}
        </div>
      </div>

      <p className="exp-narrative">{explanation.narrative}</p>

      <div className="exp-factors">
        <div className="exp-col">
          <h4 className="exp-col-title pos">▲ A favor de ingreso alto</h4>
          {explanation.positive_factors.length
            ? explanation.positive_factors.map((f, i) => (
                <FactorBar key={f.feature} factor={f} max={max} positive delay={i * 0.1} />
              ))
            : <p className="exp-empty">Ningún factor relevante a favor.</p>}
        </div>
        <div className="exp-col">
          <h4 className="exp-col-title neg">▼ En contra</h4>
          {explanation.negative_factors.length
            ? explanation.negative_factors.map((f, i) => (
                <FactorBar key={f.feature} factor={f} max={max} positive={false} delay={i * 0.1} />
              ))
            : <p className="exp-empty">Ningún factor relevante en contra.</p>}
        </div>
      </div>

      <div className="exp-footer">
        Cada barra representa el peso real que esa variable tuvo en el cálculo de la probabilidad.
        Esta explicación es 100% trazable: no es una caja negra.
      </div>
    </div>
  );
}
