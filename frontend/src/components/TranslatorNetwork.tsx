import { useState } from "react";

interface TranslatorNetworkProps {
  hoveredElement: string | null;
  setHoveredElement: (element: string | null) => void;
  onNodeClick?: (nodeId: string) => void;
  completedNodes?: string[];
}

export const TranslatorNetwork = ({
  hoveredElement,
  setHoveredElement,
  onNodeClick,
  completedNodes = [],
}: TranslatorNetworkProps) => {
  const getNodeClass = (element: string) => {
    const baseClass = "transition-all duration-300 cursor-pointer";
    if (hoveredElement === element) {
      return `${baseClass} scale-110 shadow-2xl z-10`;
    }
    return baseClass;
  };

  const getNodeStyle = (element: string) => {
    const colorMap: Record<string, string> = {
      perturbation: "#ef4444",
      drug: "#3b82f6",
      rna: "#f59e0b",
      protein: "#ec4899",
    };

    const baseColor = colorMap[element] || colorMap.perturbation;

    if (hoveredElement === element) {
      return {
        backgroundColor: baseColor,
        boxShadow: `0 0 30px ${baseColor}`,
        color: "white",
        borderColor: baseColor,
      };
    }

    return {
      backgroundColor: "white",
      borderColor: baseColor,
      color: baseColor,
    };
  };

  // Precise node positions (center coordinates in SVG space)
  const nodePositions = {
    drug: { x: 100, y: 110, radius: 48 },
    perturbation: { x: 100, y: 340, radius: 48 },
    rna: { x: 300, y: 225, radius: 55 },
    protein: { x: 500, y: 225, radius: 55 },
  };

  return (
    <div className="w-full h-full min-h-[450px] relative bg-gradient-to-br from-secondary/30 to-secondary/10 rounded-lg p-6">
      <svg viewBox="0 0 600 450" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
        {/* Arrow markers */}
        <defs>
          <marker
            id="arrowhead-inactive"
            markerWidth="12"
            markerHeight="12"
            refX="10"
            refY="4"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <polygon points="0 0, 12 4, 0 8" fill="hsl(var(--muted-foreground))" opacity="0.6" />
          </marker>
          <marker
            id="arrowhead-active"
            markerWidth="12"
            markerHeight="12"
            refX="10"
            refY="4"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <polygon points="0 0, 12 4, 0 8" fill="#10b981" />
          </marker>
        </defs>

        {/* Connections with precise positioning */}
        {/* Drug -> RNA */}
        <line
          x1={nodePositions.drug.x + nodePositions.drug.radius}
          y1={nodePositions.drug.y}
          x2={nodePositions.rna.x - nodePositions.rna.radius}
          y2={nodePositions.rna.y - 25}
          stroke={completedNodes.includes("drug-to-rna") ? "#10b981" : "hsl(var(--muted-foreground))"}
          strokeWidth={completedNodes.includes("drug-to-rna") ? "3.5" : "2.5"}
          markerEnd={completedNodes.includes("drug-to-rna") ? "url(#arrowhead-active)" : "url(#arrowhead-inactive)"}
          opacity={completedNodes.includes("drug-to-rna") ? "1" : "0.5"}
        />
        <text
          x={(nodePositions.drug.x + nodePositions.rna.x) / 2}
          y={nodePositions.drug.y - 15}
          fill={completedNodes.includes("drug-to-rna") ? "#10b981" : "hsl(var(--muted-foreground))"}
          fontSize="12"
          fontWeight="700"
          textAnchor="middle"
          className="pointer-events-none select-none"
        >
          STATE
        </text>

        {/* Gene Perturbation -> RNA */}
        <line
          x1={nodePositions.perturbation.x + nodePositions.perturbation.radius}
          y1={nodePositions.perturbation.y}
          x2={nodePositions.rna.x - nodePositions.rna.radius}
          y2={nodePositions.rna.y + 25}
          stroke={completedNodes.includes("perturb-to-rna") ? "#10b981" : "hsl(var(--muted-foreground))"}
          strokeWidth={completedNodes.includes("perturb-to-rna") ? "3.5" : "2.5"}
          markerEnd={completedNodes.includes("perturb-to-rna") ? "url(#arrowhead-active)" : "url(#arrowhead-inactive)"}
          opacity={completedNodes.includes("perturb-to-rna") ? "1" : "0.5"}
        />
        <text
          x={(nodePositions.perturbation.x + nodePositions.rna.x) / 2}
          y={nodePositions.perturbation.y + 20}
          fill={completedNodes.includes("perturb-to-rna") ? "#10b981" : "hsl(var(--muted-foreground))"}
          fontSize="12"
          fontWeight="700"
          textAnchor="middle"
          className="pointer-events-none select-none"
        >
          STATE
        </text>

        {/* RNA -> Proteomics */}
        <line
          x1={nodePositions.rna.x + nodePositions.rna.radius}
          y1={nodePositions.rna.y}
          x2={nodePositions.protein.x - nodePositions.protein.radius}
          y2={nodePositions.protein.y}
          stroke={completedNodes.includes("rna-to-protein") ? "#10b981" : "hsl(var(--muted-foreground))"}
          strokeWidth={completedNodes.includes("rna-to-protein") ? "3.5" : "2.5"}
          markerEnd={completedNodes.includes("rna-to-protein") ? "url(#arrowhead-active)" : "url(#arrowhead-inactive)"}
          opacity={completedNodes.includes("rna-to-protein") ? "1" : "0.5"}
        />
        <text
          x={(nodePositions.rna.x + nodePositions.protein.x) / 2}
          y={nodePositions.rna.y - 15}
          fill={completedNodes.includes("rna-to-protein") ? "#10b981" : "hsl(var(--muted-foreground))"}
          fontSize="12"
          fontWeight="700"
          textAnchor="middle"
          className="pointer-events-none select-none"
        >
          scTranslator
        </text>
      </svg>

      {/* Nodes with precise absolute positioning */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Drug Node */}
        <div
          className="absolute pointer-events-auto"
          style={{
            left: `${((nodePositions.drug.x - nodePositions.drug.radius) / 600) * 100}%`,
            top: `${((nodePositions.drug.y - nodePositions.drug.radius) / 450) * 100}%`,
            width: `${(nodePositions.drug.radius * 2 / 600) * 100}%`,
            paddingTop: `${(nodePositions.drug.radius * 2 / 600) * 100}%`,
          }}
          onMouseEnter={() => setHoveredElement("drug")}
          onMouseLeave={() => setHoveredElement(null)}
          onClick={() => onNodeClick?.("drug")}
        >
          <div
            className={`absolute inset-0 rounded-full border-[4px] flex flex-col items-center justify-center font-bold text-sm transition-all duration-300 ${
              completedNodes.includes("drug") ? "ring-4 ring-blue-400 ring-opacity-60" : ""
            } ${getNodeClass("drug")}`}
            style={{
              ...getNodeStyle("drug"),
              borderColor: "#3b82f6",
              color: hoveredElement === "drug" ? "white" : "#3b82f6",
              backgroundColor: hoveredElement === "drug" ? "#3b82f6" : "white",
            }}
          >
            <span className="text-center px-2 leading-tight">Drug</span>
            {completedNodes.includes("drug") && (
              <span className="text-[9px] mt-0.5 font-bold">✓ Complete</span>
            )}
          </div>
        </div>

        {/* Gene Perturbation Node */}
        <div
          className="absolute pointer-events-auto"
          style={{
            left: `${((nodePositions.perturbation.x - nodePositions.perturbation.radius) / 600) * 100}%`,
            top: `${((nodePositions.perturbation.y - nodePositions.perturbation.radius) / 450) * 100}%`,
            width: `${(nodePositions.perturbation.radius * 2 / 600) * 100}%`,
            paddingTop: `${(nodePositions.perturbation.radius * 2 / 600) * 100}%`,
          }}
          onMouseEnter={() => setHoveredElement("perturbation")}
          onMouseLeave={() => setHoveredElement(null)}
          onClick={() => onNodeClick?.("perturbation")}
        >
          <div
            className={`absolute inset-0 rounded-full border-[4px] flex flex-col items-center justify-center font-bold text-sm transition-all duration-300 ${
              completedNodes.includes("perturbation") ? "ring-4 ring-green-400 ring-opacity-60" : ""
            } ${getNodeClass("perturbation")}`}
            style={getNodeStyle("perturbation")}
          >
            <span className="text-center px-2 leading-tight">Gene Perturb</span>
            {completedNodes.includes("perturbation") && (
              <span className="text-[9px] mt-0.5 font-bold">✓ Complete</span>
            )}
          </div>
        </div>

        {/* RNA Node */}
        <div
          className="absolute pointer-events-auto"
          style={{
            left: `${((nodePositions.rna.x - nodePositions.rna.radius) / 600) * 100}%`,
            top: `${((nodePositions.rna.y - nodePositions.rna.radius) / 450) * 100}%`,
            width: `${(nodePositions.rna.radius * 2 / 600) * 100}%`,
            paddingTop: `${(nodePositions.rna.radius * 2 / 600) * 100}%`,
          }}
          onMouseEnter={() => setHoveredElement("rna")}
          onMouseLeave={() => setHoveredElement(null)}
          onClick={() => onNodeClick?.("rna")}
        >
          <div
            className={`absolute inset-0 rounded-full border-[4px] flex flex-col items-center justify-center font-bold text-sm transition-all duration-300 ${
              completedNodes.includes("rna") ? "ring-4 ring-yellow-400 ring-opacity-60" : ""
            } ${getNodeClass("rna")}`}
            style={getNodeStyle("rna")}
          >
            <span className="text-center px-2 leading-tight">RNA</span>
            {completedNodes.includes("rna") && (
              <span className="text-[9px] mt-0.5 font-bold">✓ Complete</span>
            )}
          </div>
        </div>

        {/* Proteomics Node */}
        <div
          className="absolute pointer-events-auto"
          style={{
            left: `${((nodePositions.protein.x - nodePositions.protein.radius) / 600) * 100}%`,
            top: `${((nodePositions.protein.y - nodePositions.protein.radius) / 450) * 100}%`,
            width: `${(nodePositions.protein.radius * 2 / 600) * 100}%`,
            paddingTop: `${(nodePositions.protein.radius * 2 / 600) * 100}%`,
          }}
          onMouseEnter={() => setHoveredElement("protein")}
          onMouseLeave={() => setHoveredElement(null)}
          onClick={() => onNodeClick?.("protein")}
        >
          <div
            className={`absolute inset-0 rounded-full border-[4px] flex flex-col items-center justify-center font-bold text-sm transition-all duration-300 ${
              completedNodes.includes("protein") ? "ring-4 ring-pink-400 ring-opacity-60" : ""
            } ${getNodeClass("protein")}`}
            style={getNodeStyle("protein")}
          >
            <span className="text-center px-2 leading-tight">Proteomics</span>
            {completedNodes.includes("protein") && (
              <span className="text-[9px] mt-0.5 font-bold">✓ Complete</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
