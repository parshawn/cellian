import { useState, useRef, useEffect } from "react";
import { X, Move, Maximize2, Minimize2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FloatingGraphProps {
  hoveredElement: string | null;
  setHoveredElement: (element: string | null) => void;
  isVisible: boolean;
  onClose: () => void;
  onNodeClick?: (nodeId: string) => void;
  completedNodes?: string[];
}

export const FloatingGraph = ({
  hoveredElement,
  setHoveredElement,
  isVisible,
  onClose,
  onNodeClick,
  completedNodes = [],
}: FloatingGraphProps) => {
  const [position, setPosition] = useState({ x: 20, y: 80 });
  const [isDragging, setIsDragging] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const graphRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y,
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (graphRef.current) {
      const rect = graphRef.current.getBoundingClientRect();
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
      setIsDragging(true);
    }
  };

  const getNodeClass = (element: string) => {
    const baseClass = "transition-all duration-300";
    if (hoveredElement === element) {
      return `${baseClass} scale-110 shadow-2xl`;
    }
    return baseClass;
  };

  const getNodeStyle = (element: string) => {
    const colorMap: Record<string, string> = {
      perturbation: "hsl(var(--perturbation-red))",
      drug: "hsl(var(--dna-blue))",
      nucleus: "hsl(var(--dna-blue))",
      rna: "hsl(var(--rna-yellow))",
      protein: "hsl(var(--protein-pink))",
    };

    const baseColor = colorMap[element] || colorMap.perturbation;

    if (hoveredElement === element) {
      return {
        backgroundColor: baseColor,
        boxShadow: `0 0 30px ${baseColor}`,
        color: "white",
      };
    }

    return {
      backgroundColor: "white",
      borderColor: baseColor,
      color: baseColor,
    };
  };

  if (!isVisible) return null;

  const graphSize = isExpanded ? "w-[900px] h-[550px]" : "w-[600px] h-[400px]";

  return (
    <div
      ref={graphRef}
      className={`fixed ${graphSize} bg-card/95 backdrop-blur-lg border-2 border-border rounded-xl shadow-2xl z-50 animate-fade-in`}
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between p-3 border-b border-border bg-secondary/50 rounded-t-xl cursor-move"
        onMouseDown={handleMouseDown}
      >
        <div className="flex items-center gap-2">
          <Move className="w-4 h-4 text-muted-foreground" />
          <h3 className="text-sm font-semibold text-foreground">Translator Model Network</h3>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Graph Content */}
      <div className="p-8 overflow-auto" style={{ height: "calc(100% - 52px)" }}>
        <svg viewBox="0 0 900 450" className="w-full h-full min-h-[450px]">
          {/* Edges with labels */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="12"
              markerHeight="12"
              refX="10"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 12 3.5, 0 7" fill="hsl(var(--muted-foreground))" />
            </marker>
            {completedNodes.includes("perturb-to-rna") && (
              <marker
                id="completed-arrow"
                markerWidth="12"
                markerHeight="12"
                refX="10"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 12 3.5, 0 7" fill="#10b981" />
              </marker>
            )}
          </defs>

          {/* Perturbation -> RNA */}
          <line
            x1="200"
            y1="300"
            x2="400"
            y2="200"
            stroke={completedNodes.includes("perturb-to-rna") ? "#10b981" : "hsl(var(--muted-foreground))"}
            strokeWidth={completedNodes.includes("perturb-to-rna") ? "3" : "2"}
            markerEnd={completedNodes.includes("perturb-to-rna") ? "url(#completed-arrow)" : "url(#arrowhead)"}
            opacity={completedNodes.includes("perturb-to-rna") ? "0.8" : "0.4"}
          />
          <text x="280" y="235" fill={completedNodes.includes("perturb-to-rna") ? "#10b981" : "hsl(var(--muted-foreground))"} fontSize="14" fontWeight="600">
            STATE
          </text>

          {/* RNA -> Proteomics */}
          <line
            x1="550"
            y1="200"
            x2="700"
            y2="300"
            stroke={completedNodes.includes("rna-to-protein") ? "#10b981" : "hsl(var(--muted-foreground))"}
            strokeWidth={completedNodes.includes("rna-to-protein") ? "3" : "2"}
            markerEnd={completedNodes.includes("rna-to-protein") ? "url(#completed-arrow)" : "url(#arrowhead)"}
            opacity={completedNodes.includes("rna-to-protein") ? "0.8" : "0.4"}
          />
          <text x="600" y="235" fill={completedNodes.includes("rna-to-protein") ? "#10b981" : "hsl(var(--muted-foreground))"} fontSize="14" fontWeight="600">
            scTranslator
          </text>
        </svg>

        {/* Nodes */}
        <div className="relative" style={{ marginTop: "-450px", height: "450px" }}>
          {/* Gene Perturbation Node */}
          <div
            className="absolute left-[120px] top-[250px]"
            onMouseEnter={() => setHoveredElement("perturbation")}
            onMouseLeave={() => setHoveredElement(null)}
            onClick={() => onNodeClick?.("perturbation")}
          >
            <div
              className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center cursor-pointer font-semibold text-sm transition-all duration-300 ${
                completedNodes.includes("perturbation") ? "ring-4 ring-green-500" : ""
              } ${getNodeClass("perturbation")}`}
              style={getNodeStyle("perturbation")}
            >
              <span className="text-center px-2">Gene Perturbation</span>
              {completedNodes.includes("perturbation") && (
                <span className="text-xs mt-1">✓ Complete</span>
              )}
            </div>
          </div>

          {/* Drug Node */}
          <div
            className="absolute left-[120px] top-[100px]"
            onMouseEnter={() => setHoveredElement("drug")}
            onMouseLeave={() => setHoveredElement(null)}
            onClick={() => onNodeClick?.("drug")}
          >
            <div
              className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center cursor-pointer font-semibold text-sm transition-all duration-300 ${
                completedNodes.includes("drug") ? "ring-4 ring-blue-500" : ""
              } ${getNodeClass("drug")}`}
              style={{
                ...getNodeStyle("drug"),
                borderColor: "hsl(var(--dna-blue))",
                color: hoveredElement === "drug" ? "white" : "hsl(var(--dna-blue))",
                backgroundColor: hoveredElement === "drug" ? "hsl(var(--dna-blue))" : "white",
              }}
            >
              <span className="text-center px-2">Drug</span>
              {completedNodes.includes("drug") && (
                <span className="text-xs mt-1">✓ Complete</span>
              )}
            </div>
          </div>

          {/* RNA Node */}
          <div
            className="absolute left-[360px] top-[160px]"
            onMouseEnter={() => setHoveredElement("rna")}
            onMouseLeave={() => setHoveredElement(null)}
            onClick={() => onNodeClick?.("rna")}
          >
            <div
              className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center cursor-pointer font-semibold text-sm transition-all duration-300 ${
                completedNodes.includes("rna") ? "ring-4 ring-yellow-500" : ""
              } ${getNodeClass("rna")}`}
              style={getNodeStyle("rna")}
            >
              <span className="text-center px-2">RNA</span>
              {completedNodes.includes("rna") && (
                <span className="text-xs mt-1">✓ Complete</span>
              )}
            </div>
          </div>

          {/* Proteomics Node */}
          <div
            className="absolute right-[120px] top-[250px]"
            onMouseEnter={() => setHoveredElement("protein")}
            onMouseLeave={() => setHoveredElement(null)}
            onClick={() => onNodeClick?.("protein")}
          >
            <div
              className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center cursor-pointer font-semibold text-sm transition-all duration-300 ${
                completedNodes.includes("protein") ? "ring-4 ring-pink-500" : ""
              } ${getNodeClass("protein")}`}
              style={getNodeStyle("protein")}
            >
              <span className="text-center px-2">Proteomics</span>
              {completedNodes.includes("protein") && (
                <span className="text-xs mt-1">✓ Complete</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
