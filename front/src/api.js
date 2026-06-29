import axios from "axios";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const getModels = () => axios.get(`${BASE}/models`).then((r) => r.data);

export const predict = (person, model = null) => {
  const params = model ? { model } : {};
  return axios.post(`${BASE}/predict`, person, { params }).then((r) => r.data);
};
