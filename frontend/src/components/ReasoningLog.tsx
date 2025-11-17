import { useEffect, useState } from "react";

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
  const fallbackLogEntries = getLogEntries(perturbationType);

  // Use real logs if available, otherwise use fallback
  const logEntries = logs && logs.length > 0 
    ? logs.map(log => ({ type: log.type, message: log.message }))
    : fallbackLogEntries;

  useEffect(() => {
    if (isActive) {
      if (logs && logs.length > 0) {
        // For real logs, show all immediately (they're already filtered/processed)
        setVisibleLogs(logs.length);
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
  const displayLogs = logEntries.filter((entry, index) => {
    const msg = entry.message.trim();
    // Skip empty lines
    if (!msg) {
      return false;
    }
    // Skip pure separator lines (lines that are mostly "=" characters)
    if (msg.replace(/=/g, '').trim().length === 0 && msg.length > 10) {
      return false;
    }
    // Skip very long lines that are likely data dumps
    if (entry.message.length > 200) {
      return false;
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
      <div className="space-y-2 font-mono text-xs max-h-[300px] overflow-y-auto">
        {displayLogs.slice(0, visibleLogs).map((entry, index) => (
          <div
            key={index}
            className={logs ? "" : "animate-fade-in"}
            style={logs ? {} : { animationDelay: `${index * 100}ms` }}
          >
            <span className={`font-bold ${getTypeColor(entry.type)}`}>[{entry.type}]</span>{" "}
            <span className="text-foreground/80">{entry.message}</span>
          </div>
        ))}
        {visibleLogs < displayLogs.length && isActive && !logs && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow" />
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow animation-delay-200" />
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow animation-delay-400" />
          </div>
        )}
        {logs && logs.length > 0 && isActive && (
          <div className="flex items-center gap-2 text-muted-foreground mt-2">
            <div className="w-1 h-1 bg-current rounded-full animate-pulse-glow" />
            <span className="text-xs">Streaming logs from pipeline...</span>
          </div>
        )}
      </div>
    </div>
  );
};
