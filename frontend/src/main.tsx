import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App";

const queryClient = new QueryClient();

// #region agent log
fetch("http://127.0.0.1:7835/ingest/448cfad2-0dbb-49ef-8467-dc5fbb19c120", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "ffe8cc" },
  body: JSON.stringify({
    sessionId: "ffe8cc",
    runId: "iphone-host-debug",
    hypothesisId: "H1_H2",
    location: "src/main.tsx:10",
    message: "Frontend boot reached browser",
    data: {
      href: window.location.href,
      host: window.location.host,
      hostname: window.location.hostname,
      protocol: window.location.protocol,
      userAgent: navigator.userAgent,
    },
    timestamp: Date.now(),
  }),
}).catch(() => {});
// #endregion

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
