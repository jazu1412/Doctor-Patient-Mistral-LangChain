import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import { apiFetch } from "../api/client";

type SimilarCase = {
  id: string;
  document: string;
  similarity_score: number;
};

function sanitizeAnalysisText(raw: string): string {
  return (raw || "")
    .replace(/\|\|+/g, " ")
    .replace(/^[\s|]*(?:-{3,}|={3,}|_{3,})[\s|]*$/gm, "")
    .replace(/(\s*[|]\s*){2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function ClinicalDecisionPage() {
  const [symptoms, setSymptoms] = useState("");
  const [numCases, setNumCases] = useState(5);
  const [showDetails, setShowDetails] = useState(true);

  const searchMutation = useMutation({
    mutationFn: async () => {
      const caseResult = await apiFetch<{ cases: SimilarCase[] }>(
        `/cases/similar?symptoms=${encodeURIComponent(symptoms)}&top_k=${numCases}`,
      );
      const docs = (caseResult.cases || []).slice(0, 3).map((c) => c.document);
      const analysisResult = await apiFetch<{ analysis: string }>(
        "/clinical/analysis",
        {
          method: "POST",
          body: JSON.stringify({ symptoms, case_documents: docs }),
        },
      );
      return { cases: caseResult.cases || [], analysis: analysisResult.analysis || "" };
    },
  });

  return (
    <section className="card">
      <h2>Clinical Decision Matching System</h2>
      <p className="subtitle">Search similar patient cases to support clinical decision-making.</p>
      <form
        className="form"
        onSubmit={(event) => {
          event.preventDefault();
          searchMutation.mutate();
        }}
      >
        <label>
          Enter clinical presentation:
          <textarea
            required
            rows={5}
            value={symptoms}
            onChange={(event) => setSymptoms(event.target.value)}
          />
        </label>
        <div className="row">
          <label>
            Number of similar cases
            <input
              type="number"
              min={1}
              max={10}
              value={numCases}
              onChange={(event) => setNumCases(Number(event.target.value || 5))}
            />
          </label>
          <label style={{ justifyContent: "flex-end" }}>
            Show detailed case information
            <input
              type="checkbox"
              checked={showDetails}
              onChange={(event) => setShowDetails(event.target.checked)}
            />
          </label>
        </div>
        <button className="btn primary" disabled={searchMutation.isPending}>
          {searchMutation.isPending ? "Analyzing..." : "Find Similar Cases"}
        </button>
      </form>

      {searchMutation.isError && <p className="status error">{(searchMutation.error as Error).message}</p>}

      {!!searchMutation.data?.analysis && (
        <div className="list" style={{ marginTop: 12 }}>
          {!!searchMutation.data?.cases?.length && (
            <p className="status success">Found {searchMutation.data.cases.length} similar case(s)!</p>
          )}
          <div className="listItem ai-panel">
            <strong className="sectionTitle">AI Analysis</strong>
            <div className="analysisMarkdown">
              <ReactMarkdown>{sanitizeAnalysisText(searchMutation.data.analysis)}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {!!searchMutation.data?.cases?.length && (
        <div className="list" style={{ marginTop: 12 }}>
          <strong className="sectionTitle">Case Matches</strong>
          {searchMutation.data.cases.map((row) => (
            <div key={row.id} className="listItem">
              <strong>Case: {row.id}</strong>
              <span>Similarity: {(row.similarity_score * 100).toFixed(1)}%</span>
              {showDetails && <span>{row.document}</span>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
