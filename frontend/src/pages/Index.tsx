import { useState, useEffect } from "react";
import { Cell3D } from "@/components/Cell3D";
import { TranslatorNetwork3D } from "@/components/TranslatorNetwork3D";
import { ReasoningLog } from "@/components/ReasoningLog";
import { Chatbot } from "@/components/Chatbot";
// ConditionSelector removed - condition selection is now in chatbot
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { Dashboard } from "@/components/Dashboard";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { RotateCw, BarChart3 } from "lucide-react";
import { PerturbationInfo } from "@/utils/perturbationExtractor";
import { startWorkflow, streamWorkflowResults, WorkflowStatus } from "@/utils/api";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

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
  const [activePath, setActivePath] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<"cell" | "network">("cell");
  const [perturbationInfo, setPerturbationInfo] = useState<PerturbationInfo | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus | null>(null);
  const [workflowResults, setWorkflowResults] = useState<any>(null);

  // Update graph based on pipeline stage
  useEffect(() => {
    if (!workflowStatus?.pipeline_stage) return;

    const stage = workflowStatus.pipeline_stage;
    const pertType = perturbationType;

    // Update completed nodes and active path based on pipeline stage
    if (stage === "perturbation") {
      if (pertType === "gene") {
        setCompletedNodes(["perturbation"]);
        setActivePath(["perturbation"]);
      } else {
        setCompletedNodes(["drug"]);
        setActivePath(["drug"]);
      }
    } else if (stage === "rna") {
      if (pertType === "gene") {
        setShowRNAChanges(true);
        setCompletedNodes(["perturbation", "perturb-to-rna", "rna"]);
        setActivePath(["perturbation", "perturb-to-rna", "rna"]);
      } else {
        setShowRNAChanges(true);
        setCompletedNodes(["drug", "drug-to-rna", "rna"]);
        setActivePath(["drug", "drug-to-rna", "rna"]);
      }
    } else if (stage === "protein") {
      setShowProteinChanges(true);
      if (pertType === "gene") {
        setCompletedNodes(["perturbation", "perturb-to-rna", "rna", "rna-to-protein", "protein"]);
        setActivePath(["perturbation", "perturb-to-rna", "rna", "rna-to-protein", "protein"]);
      } else {
        setCompletedNodes(["drug", "drug-to-rna", "rna", "rna-to-protein", "protein"]);
        setActivePath(["drug", "drug-to-rna", "rna", "rna-to-protein", "protein"]);
      }
    } else if (stage === "analysis") {
      setShowProteinChanges(true);
      if (pertType === "gene") {
        setCompletedNodes(["perturbation", "perturb-to-rna", "rna", "rna-to-protein", "protein"]);
        setActivePath(["perturbation", "perturb-to-rna", "rna", "rna-to-protein", "protein"]);
      } else {
        setCompletedNodes(["drug", "drug-to-rna", "rna", "rna-to-protein", "protein"]);
        setActivePath(["drug", "drug-to-rna", "rna", "rna-to-protein", "protein"]);
      }
    } else if (stage === "completed") {
      setShowProteinChanges(true);
      setHasInjected(true);
      setIsInjecting(false);
      if (pertType === "gene") {
        setCompletedNodes(["perturbation", "perturb-to-rna", "rna", "rna-to-protein", "protein"]);
        setActivePath(["perturbation", "perturb-to-rna", "rna", "rna-to-protein", "protein"]);
      } else {
        setCompletedNodes(["drug", "drug-to-rna", "rna", "rna-to-protein", "protein"]);
        setActivePath(["drug", "drug-to-rna", "rna", "rna-to-protein", "protein"]);
      }
    }
  }, [workflowStatus?.pipeline_stage, perturbationType]);

  const handleInjectionComplete = () => {
    setIsInjecting(false);
    setHasInjected(true);
  };

  const handleInjectionTrigger = (type: "gene" | "drug" | "both", extractedInfo?: PerturbationInfo) => {
    if (extractedInfo) {
      setPerturbationInfo(extractedInfo);
    }
    // Condition selection is now handled in chatbot, so we don't show the selector
    setShowConditionSelector(false);
    setHasInjected(false);
    setResultsNode(null);
    setActivePath([]);
    setCompletedNodes([]);
    setShowRNAChanges(false);
    setShowProteinChanges(false);
  };

  const [nodeDetailsOpen, setNodeDetailsOpen] = useState(false);
  const [selectedNodeDetails, setSelectedNodeDetails] = useState<{nodeId: string, title: string, content: string} | null>(null);

  const handleNodeClick = (nodeId: string) => {
    // Show details dialog for perturbation nodes
    if (nodeId === "perturbation" && perturbationInfo?.target) {
      setSelectedNodeDetails({
        nodeId: "perturbation",
        title: "Gene Perturbation",
        content: `Target Gene: ${perturbationInfo.target}\nType: ${perturbationInfo.type || "KO"}\nCondition: ${selectedCondition || "N/A"}`
      });
      setNodeDetailsOpen(true);
      return;
    }
    
    if (nodeId === "drug" && (perturbationInfo?.target2 || perturbationInfo?.type === "drug")) {
      const drugName = perturbationInfo.target2 || perturbationInfo.target;
      setSelectedNodeDetails({
        nodeId: "drug",
        title: "Drug Perturbation",
        content: `Target Drug: ${drugName}\nType: Drug perturbation\nCondition: ${selectedCondition || "N/A"}`
      });
      setNodeDetailsOpen(true);
      return;
    }
    
    // Show DEA stats for RNA/Protein nodes
    if (nodeId === "rna" && workflowResults?.pathway_analysis?.differential_rna) {
      const diffRNA = workflowResults.pathway_analysis.differential_rna;
      const count = Array.isArray(diffRNA) ? diffRNA.length : 0;
      const up = Array.isArray(diffRNA) ? diffRNA.filter((d: any) => (d.log2fc || d.log2FoldChange || 0) > 0).length : 0;
      const down = count - up;
      const significant = Array.isArray(diffRNA) ? diffRNA.filter((d: any) => (d.padj || d.padj || 1.0) < 0.05).length : 0;
      
      setSelectedNodeDetails({
        nodeId: "rna",
        title: "RNA Differential Expression",
        content: `Total Genes: ${count}\nUpregulated: ${up}\nDownregulated: ${down}\nSignificant (p-adj < 0.05): ${significant}`
      });
      setNodeDetailsOpen(true);
      return;
    }
    
    if (nodeId === "protein" && workflowResults?.pathway_analysis?.differential_protein) {
      const diffProtein = workflowResults.pathway_analysis.differential_protein;
      const count = Array.isArray(diffProtein) ? diffProtein.length : 0;
      const up = Array.isArray(diffProtein) ? diffProtein.filter((d: any) => (d.log2fc || d.log2FoldChange || 0) > 0).length : 0;
      const down = count - up;
      const significant = Array.isArray(diffProtein) ? diffProtein.filter((d: any) => (d.padj || d.padj || 1.0) < 0.05).length : 0;
      
      setSelectedNodeDetails({
        nodeId: "protein",
        title: "Protein Differential Expression",
        content: `Total Proteins: ${count}\nUpregulated: ${up}\nDownregulated: ${down}\nSignificant (p-adj < 0.05): ${significant}`
      });
      setNodeDetailsOpen(true);
      return;
    }
    
    // Fallback to old behavior
    if (nodeId === "rna" && completedNodes.includes("rna")) {
      setResultsNode("rna");
      setHoveredElement("rna");
      const path = perturbationType === "gene" 
        ? ["perturbation", "perturb-to-rna", "rna"]
        : ["drug", "drug-to-rna", "rna"];
      setActivePath(path);
    } else if (nodeId === "protein" && completedNodes.includes("protein")) {
      setResultsNode("protein");
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

  const handleConditionSelect = async (condition: Condition, pertInfo?: PerturbationInfo) => {
    setSelectedCondition(condition);
    setShowConditionSelector(false);
    
    // Use passed perturbationInfo or existing one
    const infoToUse = pertInfo || perturbationInfo;
    
    if (infoToUse && !hasInjected) {
      // Set perturbation info if passed
      if (pertInfo) {
        setPerturbationInfo(pertInfo);
      }
      const normalizeType = (value?: string | null) => (value || "").toLowerCase();
      const target2Requested = infoToUse.target2_requested || (infoToUse as any).target2Requested;
      const target2MatchInfo = infoToUse.target2_match_info || (infoToUse as any).target2MatchInfo;
      const resolvedTarget2 =
        infoToUse.target2 ||
        target2MatchInfo?.used_name ||
        target2MatchInfo?.suggested_name ||
        target2Requested ||
        null;
      
      const primaryType = normalizeType(infoToUse.type);
      const secondaryType = normalizeType(infoToUse.type2);
      const matchCandidateType = (target2MatchInfo?.candidate_type || "").toLowerCase();
      
      const primaryIsGene = ["ko", "kd", "oe"].includes(primaryType);
      const primaryIsDrug = primaryType === "drug";
      const secondaryIsGene = ["ko", "kd", "oe"].includes(secondaryType) || matchCandidateType === "gene";
      const secondaryIsDrug = secondaryType === "drug" || matchCandidateType === "drug";
      
      const hasMatchedSecond =
        Boolean(target2Requested) && Boolean(target2MatchInfo?.used_name || target2MatchInfo?.suggested_name);
      const hasBothFromLLM = infoToUse.has_both === true || hasMatchedSecond;
      
      const pipelinePerturbation: "gene" | "drug" = hasBothFromLLM
        ? "gene"
        : primaryIsGene
        ? "gene"
        : primaryIsDrug
        ? "drug"
        : secondaryIsGene
        ? "gene"
        : secondaryIsDrug
        ? "drug"
        : "gene";
      
      const workflowInfo = { ...infoToUse };
      if (!workflowInfo.target2 && resolvedTarget2) {
        workflowInfo.target2 = resolvedTarget2;
      }
      if (hasBothFromLLM) {
        workflowInfo.has_both = true;
      }
      if (!workflowInfo.type2 && matchCandidateType === "drug") {
        workflowInfo.type2 = "drug";
      }
      
      setPerturbationType(pipelinePerturbation);
      setHasBothPerturbations(hasBothFromLLM);
      setHasGeneInjected(hasBothFromLLM ? true : primaryIsGene || secondaryIsGene);
      setHasDrugInjected(hasBothFromLLM ? true : primaryIsDrug || secondaryIsDrug);
      
      // Start injection animation immediately for visual feedback - make it very visible
      setIsInjecting(true);
      
      // Set initial workflow status with a starting log
      const initMessage = hasBothFromLLM
        ? `Starting BOTH perturbations: Gene=${workflowInfo.target}, Drug=${workflowInfo.target2} (${condition} condition)...`
        : `Starting ${pipelinePerturbation} perturbation analysis for ${workflowInfo.target} (${condition} condition)...`;
      
      setWorkflowStatus({
        status: "running",
        current_step: "Initializing pipeline...",
        progress: 0.05,
        pipeline_stage: "perturbation",
        logs: [{
          type: "INIT",
          message: initMessage
        }]
      });
      
      // Start backend workflow
      try {
        const { workflow_id } = await startWorkflow(
          workflowInfo as any,
          condition || "Control",
          pipelinePerturbation
        );
        
        setWorkflowId(workflow_id);
        
        // Stream workflow results with immediate updates
        const cleanup = streamWorkflowResults(
          workflow_id,
          (status) => {
            setWorkflowStatus(status);
            
            // Update UI immediately based on status
            if (status.current_step) {
              // Update graph and cell visualizations
              if (status.current_step.includes("RNA") || status.pipeline_stage === "rna") {
                setShowRNAChanges(true);
                setCompletedNodes((prev) => {
                  const newNodes = [...prev];
                  if (!newNodes.includes("rna")) newNodes.push("rna");
                  if (pipelinePerturbation === "gene" && !newNodes.includes("perturb-to-rna")) newNodes.push("perturb-to-rna");
                  if (pipelinePerturbation === "drug" && !newNodes.includes("drug-to-rna")) newNodes.push("drug-to-rna");
                  return newNodes;
                });
              }
              
              if (status.current_step.includes("Protein") || status.pipeline_stage === "protein") {
                setShowProteinChanges(true);
                setCompletedNodes((prev) => {
                  const newNodes = [...prev];
                  if (!newNodes.includes("protein")) newNodes.push("protein");
                  if (!newNodes.includes("rna-to-protein")) newNodes.push("rna-to-protein");
                  return newNodes;
                });
              }
            }
            
            // Results are handled by useEffect based on pipeline_stage
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
        
        // Store cleanup function if needed
        // Note: In a real app, you'd use useEffect to handle cleanup on unmount
      } catch (error) {
        console.error("Failed to start workflow:", error);
        setIsInjecting(false);
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

                {/* 3D Cell Visualization */}
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

                {/* Translator Network View */}
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
                    pipelineStage={workflowStatus?.pipeline_stage || undefined}
                    currentStep={workflowStatus?.current_step || undefined}
                    perturbationInfo={perturbationInfo}
                    workflowResults={workflowResults}
                    currentPerturbationType={workflowStatus?.current_perturbation_type as "gene" | "drug" | "both" | undefined}
                  />
                  {/* Debug info - remove in production */}
                  {process.env.NODE_ENV === 'development' && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      Debug: pipeline_stage={workflowStatus?.pipeline_stage || 'null'}, 
                      current_perturbation_type={workflowStatus?.current_perturbation_type || 'null'}
                    </div>
                  )}
                </section>

                {/* Condition selection is now handled in the chatbot */}

                {/* Info Card */}
                <section className="bg-gradient-to-br from-dna/10 via-rna/10 to-protein/10 rounded-xl border border-border p-6">
                  <h3 className="text-sm font-semibold text-foreground mb-3">How It Works</h3>
                  <ol className="text-xs text-muted-foreground leading-relaxed space-y-2">
                    <li className="flex gap-2">
                      <span className="font-bold text-purple-500">1.</span>
                      <span>Ask about a gene perturbation (e.g., "What happens if I knock down TP53?" or "CHCHD2 knockout") or a drug (e.g., "Dimethyl fumarate") or both (e.g., "CHCHD2 vs Dimethyl fumarate")</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-dna">2.</span>
                      <span>Select the experimental condition (Control, IFNγ, or Co-culture) when prompted</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-rna">3.</span>
                      <span>Watch the 3D visualizations: perturbation injection → RNA prediction → protein translation → pathway analysis</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-protein">4.</span>
                      <span>View results in the Results & Dashboard tab: differential expression, pathways, and hypotheses</span>
                    </li>
                  </ol>
                </section>
              </div>

              {/* Sidebar - Controls & Reasoning */}
              <div className="space-y-6">
                {/* AI Chatbot */}
                <section>
                  <Chatbot 
                    isActive={hasInjected} 
                    onQuerySubmit={handleQuerySubmit} 
                    onInjectionTrigger={handleInjectionTrigger}
                    onConditionSelect={handleConditionSelect}
                    selectedCondition={selectedCondition}
                  />
                </section>

                {/* Reasoning Log */}
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
              perturbationType={hasBothPerturbations ? "both" : perturbationType}
              perturbationName={perturbationInfo?.target || undefined}
              perturbationName2={perturbationInfo?.target2 || undefined}
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
      
      {/* Node Details Dialog */}
      <Dialog open={nodeDetailsOpen} onOpenChange={setNodeDetailsOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{selectedNodeDetails?.title || "Node Details"}</DialogTitle>
            <DialogDescription>
              Click on any node in the 3D graph to see details
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4">
            <pre className="whitespace-pre-wrap text-sm bg-secondary p-4 rounded-lg">
              {selectedNodeDetails?.content || "No details available"}
            </pre>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Index;
