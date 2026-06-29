import axios from "axios";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE });

export const getModels    = () => api.get("/models").then(r => r.data);
export const getAnalytics = () => api.get("/analytics").then(r => r.data);
export const predict      = (person, model = null) =>
  api.post("/predict", person, { params: model ? { model } : {} }).then(r => r.data);
