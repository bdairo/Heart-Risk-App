import { useState } from "react";

type FormValues = {
  age: number;
  cholesterol: number;
  bp: number;
};

type Props = {
  onSubmit?: (values: FormValues) => Promise<void>;
};

export default function PredictionForm({ onSubmit }: Props) {
  // Example fields: adjust to your dataset!
  const [age, setAge] = useState<number | "">("");
  const [cholesterol, setCholesterol] = useState<number | "">("");
  const [bp, setBp] = useState<number | "">("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (onSubmit) {
      await onSubmit({
        age: Number(age),
        cholesterol: Number(cholesterol),
        bp: Number(bp),
      });
    }
  };

  return (
    <form onSubmit={handleSubmit} className="grid gap-4 max-w-md">
      <label className="grid gap-1">
        <span className="text-sm">Age</span>
        <input
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value === "" ? "" : Number(e.target.value))}
          className="border rounded p-2"
          required
        />
      </label>

      <label className="grid gap-1">
        <span className="text-sm">Cholesterol</span>
        <input
          type="number"
          value={cholesterol}
          onChange={(e) => setCholesterol(e.target.value === "" ? "" : Number(e.target.value))}
          className="border rounded p-2"
          required
        />
      </label>

      <label className="grid gap-1">
        <span className="text-sm">Blood Pressure (Systolic)</span>
        <input
          type="number"
          value={bp}
          onChange={(e) => setBp(e.target.value === "" ? "" : Number(e.target.value))}
          className="border rounded p-2"
          required
        />
      </label>

      <button type="submit" className="bg-black text-white rounded px-4 py-2">
        Predict
      </button>
    </form>
  );
}
