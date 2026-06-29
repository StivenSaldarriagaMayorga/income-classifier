import { useState, useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Dashboard from "./components/Dashboard";
import Classifier from "./components/Classifier";
import { getModels, getAnalytics } from "./api";
import "./App.css";

gsap.registerPlugin(ScrollTrigger);

export default function App() {
  const [tab, setTab]           = useState("classify");
  const [modelsData, setModels] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading]   = useState(true);
  const headerRef = useRef(null);

  useEffect(() => {
    Promise.all([getModels(), getAnalytics()])
      .then(([m, a]) => { setModels(m); setAnalytics(a); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!headerRef.current) return;
    gsap.fromTo(
      headerRef.current.querySelectorAll(".animate-in"),
      { y: -30, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.7, stagger: 0.12, ease: "power3.out" }
    );
  }, []);

  return (
    <div className="app">
      <header ref={headerRef} className="header">
        <div className="header-inner">
          <div className="animate-in brand">
            <span className="brand-icon">◈</span>
            <span className="brand-name">IncomeML</span>
          </div>
          <nav className="animate-in tabs">
            <button className={tab === "classify" ? "tab active" : "tab"} onClick={() => setTab("classify")}>
              Clasificador
            </button>
            <button className={tab === "dashboard" ? "tab active" : "tab"} onClick={() => setTab("dashboard")}>
              Dashboard
            </button>
          </nav>
          <div className="animate-in header-badge">
            Dataset: UCI Adult · {modelsData?.dataset?.total_samples?.toLocaleString() ?? "—"} registros
          </div>
        </div>
      </header>

      <main className="main">
        {loading ? (
          <div className="loading-screen">
            <div className="spinner" />
            <p>Cargando modelos...</p>
          </div>
        ) : tab === "classify" ? (
          <Classifier modelsData={modelsData} />
        ) : (
          <Dashboard modelsData={modelsData} analytics={analytics} />
        )}
      </main>
    </div>
  );
}
