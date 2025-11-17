import { useEffect, useState, useRef } from "react";

// LogEntry type is now imported from api.ts, but keeping local for compatibility
interface LogEntry {
  type: string;
  message: string;
  timestamp?: number;
}

interface ReasoningLogProps {
  isActive: boolean;
  perturbationType?: "gene" | "drug" | "both";
  waitingForCondition?: boolean;
  logs?: LogEntry[]; // Real-time logs from backend
}

const getLogEntries = (perturbationType: "gene" | "drug" | "both" = "gene") => {
  if (perturbationType === "gene") {
    return [
      { type: "INJECT", message: "CRISPR-Cas9 injection detected. Analyzing perturbation..." },
      { type: "INIT", message: "Initializing Multi-Omics Hypothesis Engine..." },
      { type: "PLAN", message: "Analyzing biological pathway: Gene Perturbation → RNA → Proteomics" },
      { type: "MODEL", message: "Using STATE Replogle Fine Tuned Model to get RNA from this perturbation..." },
      { type: "MODEL", message: "Using scTranslator to get protein from RNA..." },
      { type: "COMPUTE", message: "Computing multi-omic hypothesis space..." },
      { type: "VALIDATE", message: "Cross-validating predictions across modalities..." },
      { type: "RESULT", message: "Hypothesis generation complete. Confidence: 87%" },
    ];
  } else if (perturbationType === "drug") {
    return [
      { type: "INJECT", message: "Drug injection detected. Analyzing perturbation..." },
      { type: "INIT", message: "Initializing Multi-Omics Hypothesis Engine..." },
      { type: "PLAN", message: "Analyzing biological pathway: Drug Perturbation → RNA → Proteomics" },
      { type: "MODEL", message: "Using STATE Tahoe Fine Tuned Model to get RNA from this perturbation..." },
      { type: "MODEL", message: "Using scTranslator to get protein from RNA..." },
      { type: "COMPUTE", message: "Computing multi-omic hypothesis space..." },
      { type: "VALIDATE", message: "Cross-validating predictions across modalities..." },
      { type: "RESULT", message: "Hypothesis generation complete. Confidence: 87%" },
    ];
  } else {
    // Both
    return [
      { type: "INJECT", message: "CRISPR-Cas9 and Drug injection detected. Analyzing perturbations..." },
      { type: "INIT", message: "Initializing Multi-Omics Hypothesis Engine..." },
      { type: "PLAN", message: "Analyzing biological pathway: Gene & Drug Perturbations → RNA → Proteomics" },
      { type: "MODEL", message: "Using STATE Replogle Fine Tuned Model for gene perturbation..." },
      { type: "MODEL", message: "Using STATE Tahoe Fine Tuned Model for drug perturbation..." },
      { type: "MODEL", message: "Using scTranslator to get protein from RNA..." },
      { type: "COMPUTE", message: "Computing multi-omic hypothesis space..." },
      { type: "VALIDATE", message: "Cross-validating predictions across modalities..." },
      { type: "RESULT", message: "Hypothesis generation complete. Confidence: 87%" },
    ];
  }
};

export const ReasoningLog = ({ isActive, perturbationType = "gene", waitingForCondition = false, logs }: ReasoningLogProps) => {
  const [visibleLogs, setVisibleLogs] = useState<number>(0);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const fallbackLogEntries = getLogEntries(perturbationType);

  // Use real logs if available, otherwise use fallback
  // Create a new array reference to ensure React detects changes
  const logEntries = logs && logs.length > 0 
    ? logs.map(log => ({ type: log.type || "INFO", message: log.message || "" }))
    : fallbackLogEntries;
  
  // Debug: Log when logs change (remove in production)
  useEffect(() => {
    if (logs && logs.length > 0) {
      console.log(`[ReasoningLog] Received ${logs.length} logs, visibleLogs: ${visibleLogs}, displayLogs will be: ${logEntries.length}`);
    }
  }, [logs, logs?.length, visibleLogs, logEntries.length]);

  useEffect(() => {
    if (isActive) {
      if (logs && logs.length > 0) {
        // For real logs, always show all logs as they stream in
        // Update visibleLogs whenever logs array changes (new logs arrive)
        // Use the actual length to ensure we show all logs
        const logCount = logs.length;
        setVisibleLogs(logCount);
      } else {
        // For fallback logs, animate them in
        setVisibleLogs(0);
        const interval = setInterval(() => {
          setVisibleLogs((prev) => {
            if (prev < fallbackLogEntries.length) {
              return prev + 1;
            }
            return prev;
          });
        }, 800);

        return () => clearInterval(interval);
      }
    } else {
      setVisibleLogs(0);
    }
  }, [isActive, perturbationType, logs, fallbackLogEntries.length]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (logs && logs.length > 0 && scrollContainerRef.current) {
      // Use setTimeout to ensure DOM has updated
      setTimeout(() => {
        if (scrollContainerRef.current) {
          scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
      }, 100);
    }
  }, [logs]);

  const getTypeColor = (type: string) => {
    switch (type) {
      case "INJECT":
        return "text-[#a855f7]";
      case "INIT":
        return "text-dna";
      case "PLAN":
        return "text-rna";
      case "MODEL":
        return "text-protein";
      case "COMPUTE":
        return "text-perturbation";
      case "VALIDATE":
        return "text-dna";
      case "RESULT":
        return "text-foreground";
      default:
        return "text-muted-foreground";
    }
  };

  if (waitingForCondition) {
    return (
      <div className="bg-secondary/50 rounded-lg p-6 border border-border">
        <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
          <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
          Agent Reasoning
        </h3>
        <p className="text-xs text-muted-foreground">
          ⏳ Waiting for condition selection to start analysis...
        </p>
      </div>
    );
  }

  if (!isActive) {
    return (
      <div className="bg-secondary/50 rounded-lg p-6 border border-border">
        <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
          <div className="w-2 h-2 bg-muted rounded-full" />
          Agent Reasoning
        </h3>
        <p className="text-xs text-muted-foreground">
          Select a condition to start hypothesis generation...
        </p>
      </div>
    );
  }

  // Filter out very verbose or duplicate logs for display
  // For real logs, be very permissive - only filter obvious noise
  const displayLogs = logEntries.filter((entry, index) => {
    const msg = entry.message?.trim() || "";
    // Skip empty lines
    if (!msg) {
      return false;
    }
    // Skip pure separator lines (lines that are mostly "=" characters)
    if (msg.replace(/=/g, '').trim().length === 0 && msg.length > 10) {
      return false;
    }
    // For real logs, only skip extremely long lines (likely data dumps)
    if (logs && logs.length > 0) {
      // Only skip lines longer than 1000 characters for real logs
      if (entry.message.length > 1000) {
        return false;
      }
      // Always show real logs - don't filter them out
      return true;
    } else {
      // For fallback logs, use stricter filtering
      if (entry.message.length > 200) {
        return false;
      }
    }
    return true;
  });

  return (
    <div className="bg-secondary/50 rounded-lg p-6 border border-border">
      <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${isActive ? "bg-dna animate-pulse-glow" : "bg-muted"}`} />
        Agent Reasoning
        {logs && logs.length > 0 && (
          <span className="text-xs text-muted-foreground ml-2">({logs.length} logs)</span>
        )}
      </h3>
      <div 
        ref={scrollContainerRef}
        className="space-y-2 font-mono text-xs max-h-[300px] overflow-y-auto"
      >
        {displayLogs.length === 0 && logs && logs.length > 0 ? (
          // If all logs were filtered out, show them anyway (truncated)
          <div className="text-xs">
            {logEntries.slice(0, visibleLogs).map((entry, index) => (
              <div key={`raw-log-${index}`} className="mb-1">
                <span className={`font-bold ${getTypeColor(entry.type)}`}>[{entry.type}]</span>{" "}
                <span className="text-foreground/80">
                  {entry.message.length > 300 
                    ? `${entry.message.substring(0, 300)}... (truncated)` 
                    : entry.message}
                </span>
              </div>
            ))}
          </div>
        ) : (
          displayLogs.slice(0, visibleLogs).map((entry, index) => (
            <div
              key={logs ? `log-${index}-${entry.message.substring(0, 20)}` : `fallback-${index}`}
              className={logs ? "" : "animate-fade-in"}
              style={logs ? {} : { animationDelay: `${index * 100}ms` }}
            >
              <span className={`font-bold ${getTypeColor(entry.type)}`}>[{entry.type}]</span>{" "}
              <span className="text-foreground/80">{entry.message}</span>
            </div>
          ))
        )}
        {visibleLogs < displayLogs.length && isActive && !logs && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow" />
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow animation-delay-200" />
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow animation-delay-400" />
          </div>
        )}
        {logs && logs.length > 0 && isActive && displayLogs.length > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground mt-2">
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow" />
            <span className="text-xs">Streaming logs from pipeline...</span>
          </div>
        )}
      </div>
    </div>
  );
};
