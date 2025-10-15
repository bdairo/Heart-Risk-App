import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import Predict from "./pages/Predict";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        <header className="border-b bg-white">
          <nav className="mx-auto max-w-5xl px-4 py-3 flex gap-6">
            <Link to="/" className="font-semibold">XAI Dashboard</Link>
            <Link to="/predict" className="text-sm">Predict</Link>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
