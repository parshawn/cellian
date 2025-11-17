import { useState, useEffect } from "react";
import { Cell3D } from "@/components/Cell3D";
import { TranslatorNetwork3D } from "@/components/TranslatorNetwork3D";
import { ReasoningLog } from "@/components/ReasoningLog";
import { Chatbot } from "@/components/Chatbot";
import { ConditionSelector } from "@/components/ConditionSelector";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { Dashboard } from "@/components/Dashboard";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { RotateCw, BarChart3 } from "lucide-react";
import { PerturbationInfo } from "@/utils/perturbationExtractor";
import { startWorkflow, streamWorkflowResults, WorkflowStatus } from "@/utils/api";

type PerturbationType = "gene" | "drug";
type Condition = "Control" | "IFNγ" | "Co-culture" | null;

const Index = () => {
  const [hoveredElement, setHoveredElement] = useState<string | null>(null);
  const [isInjecting, setIsInjecting] = useState(false);
  const [hasInjected, setHasInjected] = useState(false);
  const [perturbationType, setPerturbationType] = useState<PerturbationType>("gene");
  const [hasBothPerturbations, setHasBothPerturbations] = useState(false);
  const [hasGeneInjected, setHasGeneInjected] = useState(false);
  const [hasDrugInjected, setHasDrugInjected] = useState(false);
  const [showConditionSelector, setShowConditionSelector] = useState(false);
  const [selectedCondition, setSelectedCondition] = useState<Condition>(null);
  const [showRNAChanges, setShowRNAChanges] = useState(false);
  const [showProteinChanges, setShowProteinChanges] = useState(false);
  const [completedNodes, setCompletedNodes] = useState<string[]>([]);
  const [resultsNode, setResultsNode] = useState<string | null>(null);
  const [activePath, setActivePath] = useState<string[]>([]); // Track active path for graph lighting
  const [viewMode, setViewMode] = useState<"cell" | "network">("cell"); // Subtab for cell vs network view
  const [perturbationInfo, setPerturbationInfo] = useState<PerturbationInfo | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus | null>(null);
  const [workflowResults, setWorkflowResults] = useState<any>(null);

  const handleInjectionComplete = () => {
    setIsInjecting(false);
    setHasInjected(true);
    
    if (hasBothPerturbations) {
      // Both gene and drug
      setHasGeneInjected(true);
      setHasDrugInjected(true);
      setCompletedNodes(["perturbation", "drug"]);
      setActivePath(["perturbation", "drug"]);
      // First: light up both perturbation → RNA edges
      setTimeout(() => {
        setShowRNAChanges(true);
        setCompletedNodes(["perturbation", "drug", "perturb-to-rna", "drug-to-rna", "rna"]);
        setActivePath(["perturbation", "drug", "perturb-to-rna", "drug-to-rna", "rna"]);
        // Then: RNA → Protein after delay
        setTimeout(() => {
          setShowProteinChanges(true);
          setCompletedNodes((prev) => [...prev, "rna-to-protein", "protein"]);
          setActivePath((prev) => [...prev, "rna-to-protein", "protein"]);
        }, 2000);
      }, 2000);
    } else if (perturbationType === "gene") {
      setHasGeneInjected(true);
      setCompletedNodes(["perturbation"]);
      setActivePath(["perturbation"]);
      // First: Gene Perturbation → RNA
      setTimeout(() => {
        setShowRNAChanges(true);
        setCompletedNodes(["perturbation", "perturb-to-rna", "rna"]);
        setActivePath(["perturbation", "perturb-to-rna", "rna"]);
        // Then: RNA → Protein after delay
        setTimeout(() => {
          setShowProteinChanges(true);
          setCompletedNodes((prev) => [...prev, "rna-to-protein", "protein"]);
          setActivePath((prev) => [...prev, "rna-to-protein", "protein"]);
        }, 2000);
      }, 2000);
    } else {
      // Drug perturbation
      setHasDrugInjected(true);
      setCompletedNodes(["drug"]);
      setActivePath(["drug"]);
      // First: Drug Perturbation → RNA
      setTimeout(() => {
        setShowRNAChanges(true);
        setCompletedNodes(["drug", "drug-to-rna", "rna"]);
        setActivePath(["drug", "drug-to-rna", "rna"]);
        // Then: RNA → Protein after delay
        setTimeout(() => {
          setShowProteinChanges(true);
          setCompletedNodes((prev) => [...prev, "rna-to-protein", "protein"]);
          setActivePath((prev) => [...prev, "rna-to-protein", "protein"]);
        }, 2000);
      }, 2000);
    }
  };

  const handleInjectionTrigger = (type: "gene" | "drug" | "both", extractedInfo?: PerturbationInfo) => {
    if (extractedInfo) {
      setPerturbationInfo(extractedInfo);
    }
    // Don't start injection yet - wait for condition selection
    // Just show condition selector
    setShowConditionSelector(true);
    // Reset state for new injection
    setHasInjected(false);
    setResultsNode(null);
    setActivePath([]);
    setCompletedNodes([]);
    setShowRNAChanges(false);
    setShowProteinChanges(false);
  };

  const handleNodeClick = (nodeId: string) => {
    if (nodeId === "rna" && completedNodes.includes("rna")) {
      setResultsNode("rna");
      // Highlight RNA components in cell
      setHoveredElement("rna");
      // Build path based on perturbation type
      const path = perturbationType === "gene" 
        ? ["perturbation", "perturb-to-rna", "rna"]
        : ["drug", "drug-to-rna", "rna"];
      setActivePath(path);
      // Simulate protein changes
      setTimeout(() => {
        setShowProteinChanges(true);
        setCompletedNodes((prev) => [...prev, "rna-to-protein", "protein"]);
        setActivePath((prev) => [...prev, "rna-to-protein", "protein"]);
      }, 1000);
    } else if (nodeId === "protein" && completedNodes.includes("protein")) {
      setResultsNode("protein");
      // Highlight protein components in cell
      setHoveredElement("protein");
      setActivePath((prev) => [...prev, "protein"]);
    } else if (nodeId === "perturbation" && completedNodes.includes("perturbation")) {
      setResultsNode("perturbation");
      setHoveredElement("perturbation");
      setActivePath(["perturbation"]);
    } else if (nodeId === "drug" && completedNodes.includes("drug")) {
      setResultsNode("drug");
      setHoveredElement("drug");
      setActivePath(["drug"]);
    }
  };

  const handleQuerySubmit = (query: string, extractedInfo?: PerturbationInfo) => {
    if (extractedInfo) {
      setPerturbationInfo(extractedInfo);
    }
    setShowConditionSelector(true);
  };

  const handleConditionSelect = async (condition: Condition) => {
    setSelectedCondition(condition);
    setShowConditionSelector(false);
    
    // Start injection and reasoning only after condition is selected
    if (perturbationInfo && !hasInjected) {
      // Determine perturbation type from extracted info
      const isGene = perturbationInfo.type === "KO" || perturbationInfo.type === "KD" || perturbationInfo.type === "OE" || 
                     (perturbationInfo.target && /^[A-Z]/.test(perturbationInfo.target));
      const isDrug = perturbationInfo.type === "drug" || 
                     (perturbationInfo.target && /^[a-z]/.test(perturbationInfo.target));
      
      const pertType: "gene" | "drug" = (isGene && isDrug) ? "gene" : (isGene ? "gene" : "drug");
      
      if (isGene && isDrug) {
        setPerturbationType("gene");
        setHasBothPerturbations(true);
      } else if (isGene) {
        setPerturbationType("gene");
        setHasBothPerturbations(false);
      } else if (isDrug) {
        setPerturbationType("drug");
        setHasBothPerturbations(false);
      }
      
      setIsInjecting(true);
      
      // Start backend workflow
      try {
        const { workflow_id } = await startWorkflow(
          perturbationInfo as any,
          condition || "Control",
          pertType
        );
        
        setWorkflowId(workflow_id);
        
        // Stream workflow results
        const cleanup = streamWorkflowResults(
          workflow_id,
          (status) => {
            setWorkflowStatus(status);
            
            // Update UI based on workflow progress
            if (status.current_step && status.current_step.includes("RNA")) {
              setShowRNAChanges(true);
              setCompletedNodes((prev) => {
                const newNodes = [...prev];
                if (!newNodes.includes("rna")) newNodes.push("rna");
                if (!newNodes.includes("perturb-to-rna")) newNodes.push("perturb-to-rna");
                return newNodes;
              });
            }
            
            if (status.current_step && status.current_step.includes("Protein")) {
              setShowProteinChanges(true);
              setCompletedNodes((prev) => {
                const newNodes = [...prev];
                if (!newNodes.includes("protein")) newNodes.push("protein");
                if (!newNodes.includes("rna-to-protein")) newNodes.push("rna-to-protein");
                return newNodes;
              });
            }
            
            if (status.status === "completed" && status.results) {
              setWorkflowResults(status.results);
              setHasInjected(true);
              setIsInjecting(false);
            }
            
            if (status.status === "error") {
              console.error("Workflow error:", status.error);
              setIsInjecting(false);
            }
          },
          (error) => {
            console.error("Workflow stream error:", error);
            setIsInjecting(false);
          }
        );
        
        // Start reasoning log immediately when workflow starts
        setIsInjecting(true);
        
        // Store cleanup function for later use if needed
        // Note: In a real app, you'd use useEffect to handle cleanup on unmount
      } catch (error) {
        console.error("Failed to start workflow:", error);
        // Fallback to local simulation
        setIsInjecting(true);
        setTimeout(() => {
          handleInjectionComplete();
        }, 2000);
      }
    }
  };


  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      {/* Header */}
      <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-40">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img 
              src="/cellian_logo.png" 
              alt="Cellian Logo" 
              className="h-10 w-auto object-contain"
            />
            <div>
              <h1 className="text-xl font-bold text-foreground">Cellian</h1>
              <p className="text-xs text-muted-foreground">Multi-Omics Hypothesis Engine</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1 rounded-full">
              Type "gene" or "drug" in the chatbot to inject
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <Tabs defaultValue="main" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="main">Cell & Network</TabsTrigger>
            <TabsTrigger value="report" disabled={!hasInjected}>
              <BarChart3 className="w-4 h-4 mr-2" />
              Report & Dashboard
            </TabsTrigger>
          </TabsList>

          <TabsContent value="main" className="mt-0">
            <div className="grid lg:grid-cols-3 gap-8">
              {/* Main Content - 3D Cell or Network */}
              <div className="lg:col-span-2 space-y-6">
                {/* Sub-tabs for Cell vs Network view */}
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={() => setViewMode("cell")}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      viewMode === "cell"
                        ? "bg-gradient-to-r from-dna to-protein text-white"
                        : "bg-secondary text-muted-foreground hover:bg-secondary/80"
                    }`}
                  >
                    Cell View
                  </button>
                  <button
                    onClick={() => setViewMode("network")}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      viewMode === "network"
                        ? "bg-gradient-to-r from-dna to-protein text-white"
                        : "bg-secondary text-muted-foreground hover:bg-secondary/80"
                    }`}
                  >
                    Translator Network
                  </button>
                </div>

                {/* 3D Cell Visualization - Always rendered but hidden when not active */}
                <section 
                  className={`bg-card/50 backdrop-blur rounded-xl border border-border p-6 shadow-lg ${viewMode === "cell" ? "block" : "hidden"}`}
                >
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                        <RotateCw className="w-5 h-5 text-dna" />
                        Interactive 3D Cellular Model
                      </h2>
                      <div className="flex items-center gap-3">
                        {hasInjected && (
                          <div className="flex items-center gap-2 bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/30">
                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse-glow" />
                            <span className="text-xs text-purple-600 dark:text-purple-400 font-medium">
                              {perturbationType === "gene" ? "CRISPR Active" : "Drug Active"}
                            </span>
                          </div>
                        )}
                        <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1 rounded-full">
                          Drag to rotate • Scroll to zoom
                        </div>
                      </div>
                    </div>
                    <Cell3D
                      hoveredElement={hoveredElement}
                      setHoveredElement={setHoveredElement}
                      isInjecting={isInjecting}
                      onInjectionComplete={handleInjectionComplete}
                      perturbationType={hasBothPerturbations ? "gene" : perturbationType}
                      showRNAChanges={showRNAChanges}
                      showProteinChanges={showProteinChanges}
                      selectedNode={resultsNode}
                      hasBothPerturbations={hasBothPerturbations}
                      hasGeneInjected={hasGeneInjected}
                      hasDrugInjected={hasDrugInjected}
                    />
                    <div className="mt-6 flex flex-wrap justify-center gap-3 text-xs">
                      <div className="flex items-center gap-2 bg-secondary/50 px-3 py-1.5 rounded-full">
                        <div className="w-3 h-3 rounded-full bg-[#5da8e8] animate-pulse-glow" />
                        <span className="text-foreground font-medium">Nucleus</span>
                      </div>
                      <div className="flex items-center gap-2 bg-secondary/50 px-3 py-1.5 rounded-full">
                        <div className="w-3 h-3 rounded-full bg-[#f59e0b] animate-pulse-glow" />
                        <span className="text-foreground font-medium">RNA</span>
                      </div>
                      <div className="flex items-center gap-2 bg-secondary/50 px-3 py-1.5 rounded-full">
                        <div className="w-3 h-3 rounded-full bg-[#ec4899] animate-pulse-glow" />
                        <span className="text-foreground font-medium">Proteins</span>
                      </div>
                      {perturbationType === "gene" ? (
                        <div className="flex items-center gap-2 bg-secondary/50 px-3 py-1.5 rounded-full">
                          <div className="w-3 h-3 rounded-full bg-[#a855f7] animate-pulse-glow" />
                          <span className="text-foreground font-medium">CRISPR-Cas9</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 bg-secondary/50 px-3 py-1.5 rounded-full">
                          <div className="w-3 h-3 rounded-full bg-[#3b82f6] animate-pulse-glow" />
                          <span className="text-foreground font-medium">Drug</span>
                        </div>
                      )}
                    </div>
                  </section>

                {/* Translator Network View - Always rendered but hidden when not active */}
                <section 
                  className={`bg-card/50 backdrop-blur rounded-xl border border-border p-6 shadow-lg ${viewMode === "network" ? "block" : "hidden"}`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                      <RotateCw className="w-5 h-5 text-rna" />
                      Translator Network
                    </h2>
                    <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1 rounded-full">
                      Drag to rotate • Scroll to zoom
                    </div>
                  </div>
                  <TranslatorNetwork3D
                    hoveredElement={hoveredElement}
                    setHoveredElement={setHoveredElement}
                    onNodeClick={handleNodeClick}
                    completedNodes={completedNodes}
                    activePath={activePath}
                  />
                </section>

                {/* Condition Selector */}
                {showConditionSelector && (
                  <section className="bg-card/80 backdrop-blur rounded-xl border border-border p-6 shadow-lg">
                    <ConditionSelector onConditionSelect={handleConditionSelect} />
                  </section>
                )}

                {/* Info Card */}
                <section className="bg-gradient-to-br from-dna/10 via-rna/10 to-protein/10 rounded-xl border border-border p-6">
                  <h3 className="text-sm font-semibold text-foreground mb-3">How It Works</h3>
                  <ol className="text-xs text-muted-foreground leading-relaxed space-y-2">
                    <li className="flex gap-2">
                      <span className="font-bold text-purple-500">1.</span>
                      <span>Type "gene" or "drug" in the chatbot to inject a perturbation into the cell</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-dna">2.</span>
                      <span>Watch as the perturbation enters the cell and affects cellular components</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-rna">3.</span>
                      <span>RNA changes are visualized when STATE translation completes - watch the graph light up</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-protein">4.</span>
                      <span>Click on nodes in the 3D translator network to see results and trigger protein translation</span>
                    </li>
                  </ol>
                </section>
              </div>

              {/* Sidebar - Controls & Reasoning */}
              <div className="space-y-6">
                {/* AI Chatbot - Main Feature */}
                <section>
                  <Chatbot isActive={hasInjected} onQuerySubmit={handleQuerySubmit} onInjectionTrigger={handleInjectionTrigger} />
                </section>

                {/* Reasoning Log - Only active after condition is selected */}
                <ReasoningLog 
                  isActive={!!selectedCondition && (isInjecting || hasInjected)} 
                  perturbationType={hasBothPerturbations ? "both" : perturbationType}
                  waitingForCondition={showConditionSelector || (perturbationInfo && !selectedCondition)}
                  logs={workflowStatus?.logs || undefined}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="report" className="mt-0">
            <Dashboard 
              completedNodes={completedNodes}
              selectedCondition={selectedCondition}
              perturbationType={perturbationType}
              perturbationName={perturbationInfo?.target || undefined}
              workflowResults={workflowResults}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Results Display */}
      {resultsNode && (
        <ResultsDisplay
          nodeId={resultsNode}
          isVisible={!!resultsNode}
          onClose={() => setResultsNode(null)}
        />
      )}
    </div>
  );
};

export default Index;
