import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ResultsDisplayProps {
  nodeId: string;
  isVisible: boolean;
  onClose: () => void;
}

export const ResultsDisplay = ({ nodeId, isVisible, onClose }: ResultsDisplayProps) => {
  if (!isVisible) return null;

  const getResults = (id: string) => {
    switch (id) {
      case "perturbation":
        return {
          title: "Gene Perturbation Results",
          data: {
            "Target Gene": "BRCA1",
            "Perturbation Type": "CRISPR Knockout",
            "Success Rate": "87%",
            "Cells Affected": "1,234",
            "Status": "Complete",
          },
          summary: "CRISPR-Cas9 successfully knocked out BRCA1 gene in 87% of target cells. RNA transcription changes detected.",
        };
      case "rna":
        return {
          title: "RNA Expression Results",
          data: {
            "Differential Genes": "234",
            "Upregulated": "156",
            "Downregulated": "78",
            "Log2 Fold Change": "+2.3 to -1.8",
            "Status": "Complete",
          },
          summary: "Significant changes in RNA expression detected. 156 genes upregulated, 78 downregulated after perturbation.",
        };
      case "protein":
        return {
          title: "Proteomics Results",
          data: {
            "Proteins Detected": "1,456",
            "Differential Proteins": "89",
            "Upregulated": "52",
            "Downregulated": "37",
            "Status": "Complete",
          },
          summary: "Proteomics analysis complete. 89 proteins show significant changes correlating with RNA expression patterns.",
        };
      default:
        return {
          title: "Results",
          data: {},
          summary: "Analysis in progress...",
        };
    }
  };

  const results = getResults(nodeId);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl bg-card border-2 border-border shadow-2xl">
        <div className="flex items-center justify-between p-6 border-b border-border">
          <h2 className="text-xl font-bold text-foreground">{results.title}</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-sm text-muted-foreground">{results.summary}</p>
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(results.data).map(([key, value]) => (
              <div key={key} className="bg-secondary/50 p-3 rounded-lg">
                <div className="text-xs text-muted-foreground font-medium">{key}</div>
                <div className="text-sm font-semibold text-foreground mt-1">{value}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="p-6 border-t border-border">
          <Button onClick={onClose} className="w-full">
            Close
          </Button>
        </div>
      </Card>
    </div>
  );
};

