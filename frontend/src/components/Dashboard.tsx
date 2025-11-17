import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer, ScatterChart, Scatter, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, TrendingDown, Activity, Target, Dna, BookOpen, Lightbulb, AlertCircle, FlaskConical } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VolcanoPlot } from "./VolcanoPlot";
import { PathwayBarChart } from "./PathwayBarChart";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

type DifferentialEntry = {
  gene: string;
  log2fc: number;
  pval: number;
  padj: number;
  direction: "up" | "down";
};

const resolvePathwayName = (path: any) => {
  return (
    path?.name ||
    path?.pathway_name ||
    path?.pathway ||
    path?.Pathway ||
    path?.term ||
    path?.Term ||
    path?.description ||
    path?.id ||
    "Pathway"
  );
};

const normalizeArray = (data: any): any[] => {
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (typeof data === "object") return Object.values(data);
  return [];
};

const parseDifferentialEntries = (rows: any): DifferentialEntry[] => {
  return normalizeArray(rows)
    .map((row: any) => {
      const log2fc = Number(row.log2fc ?? row.log2FoldChange ?? row["log2FoldChange"] ?? 0) || 0;
      const pval = Number(row.pval ?? row.pvalue ?? row["p-value"] ?? row.P_value ?? 1);
      const padj = Number(row.padj ?? row["p.adj"] ?? row.P_adj ?? row.pvalue_adj ?? row.pvalueAdj ?? 1);
      const gene = row.gene || row.Gene || row.index || row.protein;
      if (!gene) {
        return null;
      }
      const direction: "up" | "down" = log2fc >= 0 ? "up" : "down";
      return {
        gene,
        log2fc,
        pval: Number.isFinite(pval) ? pval : 1,
        padj: Number.isFinite(padj) ? padj : 1,
        direction,
      };
    })
    .filter(Boolean) as DifferentialEntry[];
};

const parsePathwayEntries = (entries: any, source: string) => {
  return normalizeArray(entries).map((p: any) => ({
    id: p.id || p.pathway_id || resolvePathwayName(p),
    name: resolvePathwayName(p),
    source,
    NES: p.NES || p.nes || p.score || p.NEScore || 0,
    FDR: p.FDR || p.fdr || p.padj || p.fdr_qval || 1.0,
    pval: p.pval || p.pvalue || p["p-value"] || p.P_value || 1.0,
    member_genes: p.member_genes || p.genes || p.leadingEdge || [],
  }));
};

interface DashboardProps {
  completedNodes: string[];
  selectedCondition: string | null;
  perturbationType: "gene" | "drug" | "both";
  perturbationName?: string;
  perturbationName2?: string; // For drug when both are present
  workflowResults?: any; // Results from backend API
  workflowResults2?: any; // Results for second perturbation if both
}

// Mock data matching backend structure
const generateDEGs = (perturbationName?: string) => [
  { gene: "CDKN1A", log2fc: -2.5, pval: 0.001, padj: 0.001, direction: "down" },
  { gene: "BAX", log2fc: -2.1, pval: 0.003, padj: 0.004, direction: "down" },
  { gene: "MDM2", log2fc: -1.8, pval: 0.005, padj: 0.006, direction: "down" },
  { gene: "P21", log2fc: -2.3, pval: 0.002, padj: 0.003, direction: "down" },
  { gene: "CCND1", log2fc: 1.5, pval: 0.01, padj: 0.012, direction: "up" },
  { gene: "CASP3", log2fc: -1.2, pval: 0.02, padj: 0.025, direction: "down" },
  { gene: "BCL2", log2fc: 1.3, pval: 0.015, padj: 0.018, direction: "up" },
  { gene: "PCNA", log2fc: 1.8, pval: 0.008, padj: 0.01, direction: "up" },
  { gene: "GADD45A", log2fc: -1.5, pval: 0.012, padj: 0.015, direction: "down" },
  { gene: "FAS", log2fc: -0.9, pval: 0.025, padj: 0.03, direction: "down" },
  { gene: "MYC", log2fc: 1.2, pval: 0.018, padj: 0.022, direction: "up" },
  { gene: "RB1", log2fc: -1.1, pval: 0.022, padj: 0.028, direction: "down" },
  { gene: "ATM", log2fc: -0.8, pval: 0.03, padj: 0.035, direction: "down" },
  { gene: "CHEK2", log2fc: -1.0, pval: 0.028, padj: 0.033, direction: "down" },
  { gene: "PTEN", log2fc: -0.7, pval: 0.035, padj: 0.04, direction: "down" },
];

const generatePathways = () => [
  { id: "KEGG_P53_PATHWAY", name: "p53 signaling pathway", source: "GSEA", NES: 2.5, FDR: 0.001, member_genes: ["TP53", "CDKN1A", "BAX", "MDM2", "P21", "ATM", "CHEK2"], genes: 7 },
  { id: "KEGG_CELL_CYCLE", name: "Cell cycle", source: "GSEA", NES: 2.1, FDR: 0.003, member_genes: ["CDKN1A", "CCND1", "RB1", "PCNA", "MYC"], genes: 5 },
  { id: "KEGG_APOPTOSIS", name: "Apoptosis", source: "GSEA", NES: -2.3, FDR: 0.002, member_genes: ["BAX", "BCL2", "CASP3", "FAS"], genes: 4 },
  { id: "KEGG_DNA_REPAIR", name: "DNA repair", source: "PSEA", NES: -1.8, FDR: 0.01, member_genes: ["ATM", "CHEK2", "GADD45A", "TP53"], genes: 4 },
  { id: "REACTOME_CELL_CYCLE_CHECKPOINTS", name: "Cell cycle checkpoints", source: "GSEA", NES: 1.9, FDR: 0.008, member_genes: ["TP53", "CDKN1A", "RB1", "ATM"], genes: 4 },
  { id: "GO_CELL_CYCLE_ARREST", name: "Cell cycle arrest", source: "GO", NES: 2.0, FDR: 0.005, member_genes: ["CDKN1A", "P21", "RB1"], genes: 3 },
];

const generatePhenotypes = () => [
  { phenotype_id: "PHENO_001", name: "Cell cycle arrest", score: 0.85, direction: "increase", confidence: 0.9 },
  { phenotype_id: "PHENO_002", name: "Apoptosis resistance", score: 0.72, direction: "increase", confidence: 0.85 },
  { phenotype_id: "PHENO_003", name: "DNA damage response", score: 0.68, direction: "increase", confidence: 0.8 },
  { phenotype_id: "PHENO_004", name: "Cell proliferation", score: -0.65, direction: "decrease", confidence: 0.75 },
  { phenotype_id: "PHENO_005", name: "Senescence", score: 0.58, direction: "increase", confidence: 0.7 },
];

const generateHypotheses = () => [
  {
    id: "H1",
    statement: "TP53 knockout increases cell cycle arrest in HepG2 cells",
    mechanism: [
      "Perturbation: TP53 KO",
      "Affects genes: CDKN1A, P21, RB1",
      "Enriches pathway: p53 signaling pathway",
      "Predicts phenotype: Cell cycle arrest (increase)"
    ],
    phenotype_support: {
      primary_phenotype: "Cell cycle arrest",
      score: 0.85,
      direction: "increase",
      supporting_genes: ["CDKN1A", "P21", "RB1", "CCND1"],
      supporting_pathways: ["p53 signaling pathway", "Cell cycle"]
    },
    predicted_readouts: ["Increased CDKN1A expression", "Decreased CCND1 expression", "G1/S checkpoint activation"],
    literature_support: {
      support_level: "strong",
      papers: 12,
      pmids: ["PMID:12345678", "PMID:23456789"]
    },
    experiments: ["Flow cytometry for cell cycle analysis", "Western blot for CDKN1A", "qPCR for checkpoint genes"]
  },
  {
    id: "H2",
    statement: "TP53 knockout decreases apoptosis through BAX/BCL2 regulation",
    mechanism: [
      "Perturbation: TP53 KO",
      "Affects genes: BAX, BCL2, CASP3",
      "Enriches pathway: Apoptosis",
      "Predicts phenotype: Apoptosis resistance (increase)"
    ],
    phenotype_support: {
      primary_phenotype: "Apoptosis resistance",
      score: 0.72,
      direction: "increase",
      supporting_genes: ["BAX", "BCL2", "CASP3", "FAS"],
      supporting_pathways: ["Apoptosis"]
    },
    predicted_readouts: ["Decreased BAX expression", "Increased BCL2 expression", "Reduced caspase-3 activity"],
    literature_support: {
      support_level: "moderate",
      papers: 8,
      pmids: ["PMID:34567890"]
    },
    experiments: ["Annexin V/PI staining", "Caspase-3 activity assay", "BCL2/BAX ratio measurement"]
  },
];

// Volcano plot data
const getVolcanoData = (degs: Array<{gene: string; log2fc: number; pval: number; padj: number; direction: "up" | "down"}>) => {
  return degs.map(d => ({
    gene: d.gene,
    x: d.log2fc,
    y: -Math.log10(d.pval),
    pval: d.pval,
    padj: d.padj,
    log2fc: d.log2fc,
    direction: d.direction,
    significant: d.pval < 0.05,
  }));
};

export const Dashboard = ({ completedNodes, selectedCondition, perturbationType, perturbationName, perturbationName2, workflowResults }: DashboardProps) => {
  const [resultView, setResultView] = useState<"gene" | "drug">("gene");
  // Extract real data from workflowResults if available, otherwise use mock data
  let degs: DifferentialEntry[] = [];
  let degsDrug: DifferentialEntry[] = [];
  let genePathways: any[] = [];
  let drugPathways: any[] = [];
  let phenotypes: any[] = [];
  let hypotheses: any[] = [];
  let rnaMetrics: any = {};
  let proteinMetrics: any = {};
  let proteinDE: DifferentialEntry[] = [];
  let proteinDEDrug: DifferentialEntry[] = [];
  let proteinPathwaysGene: any[] = [];
  let proteinPathwaysDrug: any[] = [];
  
  // Debug: Log what we're receiving
  if (workflowResults) {
    console.log("Dashboard: Received workflowResults:", {
      hasPathwayAnalysis: !!workflowResults.pathway_analysis,
      pathwayKeys: workflowResults.pathway_analysis ? Object.keys(workflowResults.pathway_analysis) : [],
      hasRNAMetrics: !!workflowResults.rna_metrics,
      hasProteinMetrics: !!workflowResults.protein_metrics,
    });
  } else {
    console.log("Dashboard: No workflowResults, using mock data");
  }
  
  if (workflowResults) {
    const pathwayAnalysis = workflowResults.pathway_analysis || {};
    
    degs = parseDifferentialEntries(pathwayAnalysis.differential_rna);
    degsDrug = parseDifferentialEntries(
      pathwayAnalysis.differential_rna_drug || pathwayAnalysis.differential_rna_drug?.data
    );
    
    const gseaRNA = pathwayAnalysis.gsea_rna;
    const gseaRNADrug = pathwayAnalysis.gsea_rna_drug;
    const keggEnrichment = pathwayAnalysis.kegg_enrichment;
    const keggEnrichmentDrug = pathwayAnalysis.kegg_enrichment_drug;
    const reactomeEnrichment = pathwayAnalysis.reactome_enrichment;
    const reactomeEnrichmentDrug = pathwayAnalysis.reactome_enrichment_drug;
    const goEnrichment = pathwayAnalysis.go_enrichment;
    const goEnrichmentDrug = pathwayAnalysis.go_enrichment_drug;
    
    genePathways = [];
    drugPathways = [];
    
    const pushGenePathways = (entries: any, source: string) => {
      genePathways.push(...parsePathwayEntries(entries, source));
    };
    const pushDrugPathways = (entries: any, source: string) => {
      drugPathways.push(...parsePathwayEntries(entries, source));
    };
    
    if (gseaRNA) pushGenePathways(gseaRNA, "GSEA");
    if (keggEnrichment) pushGenePathways(keggEnrichment, "KEGG");
    if (reactomeEnrichment) pushGenePathways(reactomeEnrichment, "Reactome");
    if (goEnrichment) pushGenePathways(goEnrichment, "GO");
    
    if (gseaRNADrug) pushDrugPathways(gseaRNADrug, "GSEA");
    if (keggEnrichmentDrug) pushDrugPathways(keggEnrichmentDrug, "KEGG");
    if (reactomeEnrichmentDrug) pushDrugPathways(reactomeEnrichmentDrug, "Reactome");
    if (goEnrichmentDrug) pushDrugPathways(goEnrichmentDrug, "GO");
    
    rnaMetrics = workflowResults.rna_metrics || {};
    proteinMetrics = workflowResults.protein_metrics || {};
    
    if (workflowResults.hypotheses && Array.isArray(workflowResults.hypotheses)) {
      hypotheses = workflowResults.hypotheses;
    }
    
    if (workflowResults.phenotypes && Array.isArray(workflowResults.phenotypes)) {
      phenotypes = workflowResults.phenotypes;
    }
    
    console.log("Dashboard: Extracted data:", {
      degsCount: degs.length,
      drugDegsCount: degsDrug.length,
      pathwaysCount: genePathways.length,
      drugPathwaysCount: drugPathways.length,
      phenotypesCount: phenotypes.length,
      hypothesesCount: hypotheses.length,
      hasRNAMetrics: Object.keys(rnaMetrics).length > 0,
      hasProteinMetrics: Object.keys(proteinMetrics).length > 0,
    });
  }
  
  // Track which data is pending (not using mock data anymore)
  const dataPending = {
    degs: degs.length === 0 && degsDrug.length === 0,
    pathways: genePathways.length === 0 && drugPathways.length === 0,
    phenotypes: phenotypes.length === 0,
    hypotheses: hypotheses.length === 0,
  };
  
if (workflowResults) {
  const pathwayAnalysis = workflowResults.pathway_analysis || {};
  proteinDE = parseDifferentialEntries(pathwayAnalysis.differential_protein);
  proteinDEDrug = parseDifferentialEntries(pathwayAnalysis.differential_protein_drug);
  
  const gseaProtein = pathwayAnalysis.gsea_protein;
  const gseaProteinDrug = pathwayAnalysis.gsea_protein_drug;
  const keggProtein = pathwayAnalysis.kegg_enrichment;
  const keggProteinDrug = pathwayAnalysis.kegg_enrichment_drug;
  const reactomeProtein = pathwayAnalysis.reactome_enrichment;
  const reactomeProteinDrug = pathwayAnalysis.reactome_enrichment_drug;
  
  proteinPathwaysGene = [];
  proteinPathwaysDrug = [];
  
  if (gseaProtein) proteinPathwaysGene.push(...parsePathwayEntries(gseaProtein, "GSEA"));
  if (keggProtein) proteinPathwaysGene.push(...parsePathwayEntries(keggProtein, "KEGG"));
  if (reactomeProtein) proteinPathwaysGene.push(...parsePathwayEntries(reactomeProtein, "Reactome"));
  
  if (gseaProteinDrug) proteinPathwaysDrug.push(...parsePathwayEntries(gseaProteinDrug, "GSEA"));
  if (keggProteinDrug) proteinPathwaysDrug.push(...parsePathwayEntries(keggProteinDrug, "KEGG"));
  if (reactomeProteinDrug) proteinPathwaysDrug.push(...parsePathwayEntries(reactomeProteinDrug, "Reactome"));
}

const proteinStageComplete = completedNodes.includes("protein");
const proteinHasResults =
  proteinDE.length > 0 ||
  proteinDEDrug.length > 0 ||
  proteinPathwaysGene.length > 0 ||
  proteinPathwaysDrug.length > 0;
  
const hasDrugResults =
  degsDrug.length > 0 ||
  drugPathways.length > 0 ||
  proteinDEDrug.length > 0 ||
  proteinPathwaysDrug.length > 0;

const useDrugData =
  hasDrugResults &&
  resultView === "drug" &&
  (degsDrug.length > 0 || drugPathways.length > 0 || proteinDEDrug.length > 0 || proteinPathwaysDrug.length > 0);

useEffect(() => {
  if (!hasDrugResults && resultView !== "gene") {
    setResultView("gene");
  }
}, [hasDrugResults, resultView]);

const currentRNADE =
  useDrugData ? (degsDrug.length > 0 ? degsDrug : degs) : (degs.length > 0 ? degs : degsDrug);
const currentPathways =
  useDrugData ? (drugPathways.length > 0 ? drugPathways : genePathways) : (genePathways.length > 0 ? genePathways : drugPathways);
const currentProteinDE =
  useDrugData ? (proteinDEDrug.length > 0 ? proteinDEDrug : proteinDE) : (proteinDE.length > 0 ? proteinDE : proteinDEDrug);
const currentProteinPathways =
  useDrugData
    ? (proteinPathwaysDrug.length > 0 ? proteinPathwaysDrug : proteinPathwaysGene)
    : (proteinPathwaysGene.length > 0 ? proteinPathwaysGene : proteinPathwaysDrug);

const currentViewLabel = useDrugData ? "Drug Perturbation" : "Gene Perturbation";

const volcanoDataRNA = getVolcanoData(currentRNADE);
const volcanoDataProtein = getVolcanoData(currentProteinDE);

const upregulatedRNA = currentRNADE.filter(d => d.direction === "up");
const downregulatedRNA = currentRNADE.filter(d => d.direction === "down");
const significantRNA = currentRNADE.filter(d => d.pval < 0.05);

const upregulatedProtein = currentProteinDE.filter(d => d.direction === "up");
const downregulatedProtein = currentProteinDE.filter(d => d.direction === "down");
const significantProtein = currentProteinDE.filter(d => d.pval < 0.05);
  
  const formatNumber = (value: number, digits = 2) =>
    Number.isFinite(value) ? value.toFixed(digits) : "—";
  const isSignificant = (value: number) => Number.isFinite(value) && value < 0.05;

  // Determine perturbation label
  const perturbationLabel = perturbationType === "both" 
    ? `${perturbationName} (gene) + ${perturbationName2 || "drug"} (drug)`
    : perturbationType === "gene"
    ? `${perturbationName} (gene ${workflowResults?.target_gene || ""})`
    : `${perturbationName} (drug)`;

  return (
    <div className="space-y-6 p-6">
      {/* Perturbation Info Header */}
      <Card className="p-4 bg-gradient-to-r from-dna/10 to-protein/10 border-dna/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Perturbation Analysis</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {perturbationLabel} | Condition: {selectedCondition || "N/A"}
            </p>
          </div>
          <Badge variant="outline" className="text-sm">
            {perturbationType === "both" ? "Gene + Drug" : perturbationType === "gene" ? "Gene" : "Drug"}
          </Badge>
        </div>
      </Card>
      
      {hasDrugResults && (
        <div className="flex justify-end">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Viewing:</span>
            <Select value={resultView} onValueChange={(value) => setResultView(value as "gene" | "drug")}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Select view" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gene">Gene perturbation</SelectItem>
                <SelectItem value="drug">Drug perturbation</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )}
      
      {/* Main Tabs: RNA and Protein */}
      <Tabs defaultValue="rna" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="rna" className="flex items-center gap-2">
            <Dna className="w-4 h-4" />
            RNA Analysis
          </TabsTrigger>
          <TabsTrigger value="protein" className="flex items-center gap-2">
            <FlaskConical className="w-4 h-4" />
            Protein Analysis
          </TabsTrigger>
        </TabsList>
        
        {/* RNA Tab */}
        <TabsContent value="rna" className="space-y-6 mt-6">
          {completedNodes.includes("rna") ? (
            <>
              {/* Summary Cards for RNA */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4 bg-gradient-to-br from-rna/10 to-rna/5 border-rna/20">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">{currentViewLabel} Differential Genes</p>
                      <p className="text-2xl font-bold text-rna">{significantRNA.length}</p>
                      <p className="text-xs text-muted-foreground mt-1">p-value &lt; 0.05</p>
                    </div>
                    <Activity className="w-8 h-8 text-rna" />
                  </div>
                </Card>
                <Card className="p-4 bg-gradient-to-br from-green-500/10 to-green-500/5 border-green-500/20">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Upregulated</p>
                      <p className="text-2xl font-bold text-green-600">{upregulatedRNA.length}</p>
                    </div>
                    <TrendingUp className="w-8 h-8 text-green-600" />
                  </div>
                </Card>
                <Card className="p-4 bg-gradient-to-br from-red-500/10 to-red-500/5 border-red-500/20">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Downregulated</p>
                      <p className="text-2xl font-bold text-red-600">{downregulatedRNA.length}</p>
                    </div>
                    <TrendingDown className="w-8 h-8 text-red-600" />
                  </div>
                </Card>
                <Card className="p-4 bg-gradient-to-br from-protein/10 to-protein/5 border-protein/20">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Pathways</p>
                      <p className="text-2xl font-bold text-protein">{currentPathways.filter(p => (p.pval ?? p.FDR ?? 1) < 0.05).length}</p>
                      <p className="text-xs text-muted-foreground mt-1">p-value &lt; 0.05</p>
                    </div>
                    <Target className="w-8 h-8 text-protein" />
                  </div>
                </Card>
              </div>

              {/* Volcano Plot for RNA */}
              <VolcanoPlot 
                data={volcanoDataRNA} 
                title={`${currentViewLabel} RNA Differential Expression - Volcano Plot`}
                xLabel="log₂ Fold Change"
                yLabel="-log₁₀(p-value)"
              />

              {/* Pathway Analysis for RNA */}
              {currentPathways.length > 0 && (
                <PathwayBarChart 
                  data={currentPathways} 
                  title={`${currentViewLabel} RNA Pathway Enrichment Analysis`}
                  maxItems={20}
                />
              )}

              {/* DEGs Table */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Top Differential Genes — {currentViewLabel}</h3>
                <ScrollArea className="h-[400px]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-secondary">
                      <tr>
                        <th className="text-left p-2 font-semibold">Gene</th>
                        <th className="text-right p-2 font-semibold">Log2FC</th>
                        <th className="text-right p-2 font-semibold">p-value</th>
                        <th className="text-right p-2 font-semibold">p-adj</th>
                        <th className="text-center p-2 font-semibold">Direction</th>
                        <th className="text-center p-2 font-semibold">Significance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentRNADE
                        .sort((a, b) => Math.abs(b.log2fc) - Math.abs(a.log2fc))
                        .slice(0, 50)
                        .map((deg, i) => (
                          <tr key={i} className="border-b border-border hover:bg-secondary/50">
                            <td className="p-2 font-medium">{deg.gene}</td>
                            <td className="p-2 text-right">{formatNumber(deg.log2fc, 2)}</td>
                            <td className="p-2 text-right">{formatNumber(deg.pval, 4)}</td>
                            <td className="p-2 text-right">{formatNumber(deg.padj, 4)}</td>
                            <td className="p-2 text-center">
                              <Badge variant={deg.direction === "up" ? "default" : "destructive"}>
                                {deg.direction === "up" ? "↑ Up" : "↓ Down"}
                              </Badge>
                            </td>
                            <td className="p-2 text-center">
                              {isSignificant(deg.pval) ? (
                                <Badge variant="default">p &lt; 0.05</Badge>
                              ) : (
                                <span className="text-xs text-muted-foreground">n.s.</span>
                              )}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </ScrollArea>
              </Card>
            </>
          ) : (
            <Card className="p-6">
              <p className="text-muted-foreground">RNA analysis not yet completed. Waiting for pipeline...</p>
            </Card>
          )}
        </TabsContent>
        
        {/* Protein Tab */}
        <TabsContent value="protein" className="space-y-6 mt-6">
          {proteinStageComplete ? (
            proteinHasResults ? (
              <>
                {/* Summary Cards for Protein */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="p-4 bg-gradient-to-br from-protein/10 to-protein/5 border-protein/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">{currentViewLabel} Differential Proteins</p>
                        <p className="text-2xl font-bold text-protein">{significantProtein.length}</p>
                        <p className="text-xs text-muted-foreground mt-1">p-value &lt; 0.05</p>
                      </div>
                      <FlaskConical className="w-8 h-8 text-protein" />
                    </div>
                  </Card>
                  <Card className="p-4 bg-gradient-to-br from-green-500/10 to-green-500/5 border-green-500/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Upregulated</p>
                        <p className="text-2xl font-bold text-green-600">{upregulatedProtein.length}</p>
                      </div>
                      <TrendingUp className="w-8 h-8 text-green-600" />
                    </div>
                  </Card>
                  <Card className="p-4 bg-gradient-to-br from-red-500/10 to-red-500/5 border-red-500/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Downregulated</p>
                        <p className="text-2xl font-bold text-red-600">{downregulatedProtein.length}</p>
                      </div>
                      <TrendingDown className="w-8 h-8 text-red-600" />
                    </div>
                  </Card>
                </div>

                {/* Volcano Plot for Protein */}
                {currentProteinDE.length > 0 ? (
                  <VolcanoPlot 
                    data={volcanoDataProtein} 
                    title={`${currentViewLabel} Protein Differential Expression - Volcano Plot`}
                    xLabel="log₂ Fold Change"
                    yLabel="-log₁₀(p-value)"
                  />
                ) : (
                  <Card className="p-6">
                    <p className="text-muted-foreground">No differential protein signals available for this view.</p>
                  </Card>
                )}

                {/* Pathway Analysis for Protein */}
                {currentProteinPathways.length > 0 ? (
                  <PathwayBarChart 
                    data={currentProteinPathways} 
                    title={`${currentViewLabel} Protein Pathway Enrichment Analysis`}
                    maxItems={20}
                  />
                ) : (
                  <Card className="p-6">
                    <p className="text-muted-foreground">No enriched protein pathways detected for this view.</p>
                  </Card>
                )}

                {/* Protein DE Table */}
                {currentProteinDE.length > 0 ? (
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Top Differential Proteins — {currentViewLabel}</h3>
                    <ScrollArea className="h-[400px]">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-secondary">
                          <tr>
                            <th className="text-left p-2 font-semibold">Protein</th>
                            <th className="text-right p-2 font-semibold">Log2FC</th>
                            <th className="text-right p-2 font-semibold">p-value</th>
                            <th className="text-right p-2 font-semibold">p-adj</th>
                            <th className="text-center p-2 font-semibold">Direction</th>
                            <th className="text-center p-2 font-semibold">Significance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {currentProteinDE
                            .sort((a, b) => Math.abs(b.log2fc) - Math.abs(a.log2fc))
                            .slice(0, 50)
                            .map((deg, i) => (
                              <tr key={i} className="border-b border-border hover:bg-secondary/50">
                                <td className="p-2 font-medium">{deg.gene}</td>
                                <td className="p-2 text-right">{formatNumber(deg.log2fc, 2)}</td>
                                <td className="p-2 text-right">{formatNumber(deg.pval, 4)}</td>
                                <td className="p-2 text-right">{formatNumber(deg.padj, 4)}</td>
                                <td className="p-2 text-center">
                                  <Badge variant={deg.direction === "up" ? "default" : "destructive"}>
                                    {deg.direction === "up" ? "↑ Up" : "↓ Down"}
                                  </Badge>
                                </td>
                                <td className="p-2 text-center">
                                  {isSignificant(deg.pval) ? (
                                    <Badge variant="default">p &lt; 0.05</Badge>
                                  ) : (
                                    <span className="text-xs text-muted-foreground">n.s.</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </ScrollArea>
                  </Card>
                ) : (
                  <Card className="p-6">
                    <p className="text-muted-foreground">Result pending... Differential expression data not yet available.</p>
                  </Card>
                )}
              </>
            ) : (
              <Card className="p-6">
                <div className="flex flex-col gap-2 text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-protein animate-pulse" />
                    <p>No significant protein changes detected (p-value ≥ 0.05).</p>
                  </div>
                  <p className="text-xs">
                    Live logs will continue streaming in the reasoning panel even after analysis completes.
                  </p>
                </div>
              </Card>
            )
          ) : (
            <Card className="p-6">
              <p className="text-muted-foreground">Protein analysis not yet completed. Waiting for pipeline...</p>
            </Card>
          )}
        </TabsContent>
      </Tabs>
      
      {/* Hypotheses Section - Below RNA/Protein tabs */}
      {completedNodes.includes("protein") && (
        <Card className="p-6 mt-6">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb className="w-5 h-5 text-yellow-500" />
            <h2 className="text-xl font-bold">Generated Hypotheses</h2>
          </div>
          {hypotheses.length > 0 ? (
            <div className="space-y-4">
              {hypotheses.map((hyp: any, i: number) => (
                <Card key={i} className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold">{hyp.id}: {hyp.statement}</h3>
                    <Badge variant={hyp.literature_support?.support_level === "strong" ? "default" : "secondary"}>
                      {hyp.literature_support?.support_level || "moderate"} support
                    </Badge>
                  </div>
                  {hyp.mechanism && Array.isArray(hyp.mechanism) && (
                    <ul className="list-disc list-inside text-sm text-muted-foreground mt-2">
                      {hyp.mechanism.map((step: string, j: number) => (
                        <li key={j}>{step}</li>
                      ))}
                    </ul>
                  )}
                </Card>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Activity className="w-5 h-5 animate-pulse" />
              <p>Result pending... Hypotheses generation is still processing.</p>
            </div>
          )}
        </Card>
      )}
    </div>
  );
};
