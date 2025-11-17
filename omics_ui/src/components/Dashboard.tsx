import { Card } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer, ScatterChart, Scatter, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, TrendingDown, Activity, Target, Dna, BookOpen, Lightbulb, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DashboardProps {
  completedNodes: string[];
  selectedCondition: string | null;
  perturbationType: "gene" | "drug";
  perturbationName?: string;
  workflowResults?: any; // Results from backend API
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
const getVolcanoData = (degs: Array<{gene: string; log2fc: number; pval: number; padj: number; direction: string}>) => {
  return degs.map(d => ({
    gene: d.gene,
    x: d.log2fc,
    y: -Math.log10(d.pval),
    pval: d.pval,
    log2fc: d.log2fc,
    direction: d.direction,
    significant: d.padj < 0.05,
  }));
};

export const Dashboard = ({ completedNodes, selectedCondition, perturbationType, perturbationName, workflowResults }: DashboardProps) => {
  // Extract real data from workflowResults if available, otherwise use mock data
  let degs: any[] = [];
  let pathways: any[] = [];
  let phenotypes: any[] = [];
  let hypotheses: any[] = [];
  let rnaMetrics: any = {};
  let proteinMetrics: any = {};
  
  if (workflowResults) {
    // Extract DEGs from pathway_analysis
    const pathwayAnalysis = workflowResults.pathway_analysis || {};
    const diffRNA = pathwayAnalysis.differential_rna;
    const diffProtein = pathwayAnalysis.differential_protein;
    
    if (diffRNA && Array.isArray(diffRNA)) {
      degs = diffRNA.map((row: any) => ({
        gene: row.gene || row.Gene || row.index,
        log2fc: row.log2fc || row.log2FoldChange || row['log2FoldChange'] || 0,
        pval: row.pval || row.pvalue || row['p-value'] || row.P_value || 1.0,
        padj: row.padj || row.padj || row['p.adj'] || row.P_adj || 1.0,
        direction: (row.log2fc || row.log2FoldChange || 0) > 0 ? "up" : "down"
      }));
    } else if (diffRNA && typeof diffRNA === 'object' && 'to_dict' in diffRNA) {
      // Handle pandas DataFrame-like object
      try {
        const rows = Object.values(diffRNA);
        degs = rows.map((row: any) => ({
          gene: row.gene || row.Gene || row.index,
          log2fc: row.log2fc || row.log2FoldChange || 0,
          pval: row.pval || row.pvalue || 1.0,
          padj: row.padj || row.padj || 1.0,
          direction: (row.log2fc || 0) > 0 ? "up" : "down"
        }));
      } catch (e) {
        console.warn("Could not parse differential RNA data:", e);
      }
    }
    
    // Extract pathways from GSEA and enrichment results
    const gseaRNA = pathwayAnalysis.gsea_rna;
    const keggEnrichment = pathwayAnalysis.kegg_enrichment;
    const reactomeEnrichment = pathwayAnalysis.reactome_enrichment;
    const goEnrichment = pathwayAnalysis.go_enrichment;
    
    pathways = [];
    if (gseaRNA && Array.isArray(gseaRNA)) {
      pathways.push(...gseaRNA.map((p: any) => ({
        id: p.id || p.pathway_id || p.name,
        name: p.name || p.pathway_name || p.id,
        source: "GSEA",
        NES: p.NES || p.nes || 0,
        FDR: p.FDR || p.fdr || p.padj || 1.0,
        pval: p.pval || p.pvalue || p['p-value'] || 1.0,
        member_genes: p.member_genes || p.genes || []
      })));
    }
    
    if (keggEnrichment && Array.isArray(keggEnrichment)) {
      pathways.push(...keggEnrichment.map((p: any) => ({
        id: p.id || p.pathway_id || p.name,
        name: p.name || p.pathway_name || p.id,
        source: "KEGG",
        NES: p.NES || p.nes || 0,
        FDR: p.FDR || p.fdr || p.padj || 1.0,
        pval: p.pval || p.pvalue || p['p-value'] || 1.0,
        member_genes: p.member_genes || p.genes || []
      })));
    }
    
    // Extract metrics
    rnaMetrics = workflowResults.rna_metrics || {};
    proteinMetrics = workflowResults.protein_metrics || {};
  }
  
  // Fallback to mock data if no real data
  if (degs.length === 0) {
    degs = generateDEGs(perturbationName);
  }
  if (pathways.length === 0) {
    pathways = generatePathways();
  }
  if (phenotypes.length === 0) {
    phenotypes = generatePhenotypes();
  }
  if (hypotheses.length === 0) {
    hypotheses = generateHypotheses();
  }
  
  const volcanoData = getVolcanoData(degs);
  
  const upregulated = degs.filter(d => d.direction === "up");
  const downregulated = degs.filter(d => d.direction === "down");
  const significant = degs.filter(d => d.padj < 0.05);

  return (
    <div className="space-y-6 p-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4 bg-gradient-to-br from-rna/10 to-rna/5 border-rna/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Differential Genes</p>
              <p className="text-2xl font-bold text-rna">{significant.length}</p>
              <p className="text-xs text-muted-foreground mt-1">p-adj &lt; 0.05</p>
            </div>
            <Activity className="w-8 h-8 text-rna" />
          </div>
        </Card>
        <Card className="p-4 bg-gradient-to-br from-green-500/10 to-green-500/5 border-green-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Upregulated</p>
              <p className="text-2xl font-bold text-green-600">{upregulated.length}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-600" />
          </div>
        </Card>
        <Card className="p-4 bg-gradient-to-br from-red-500/10 to-red-500/5 border-red-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Downregulated</p>
              <p className="text-2xl font-bold text-red-600">{downregulated.length}</p>
            </div>
            <TrendingDown className="w-8 h-8 text-red-600" />
          </div>
        </Card>
        <Card className="p-4 bg-gradient-to-br from-protein/10 to-protein/5 border-protein/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Pathways</p>
              <p className="text-2xl font-bold text-protein">{pathways.filter(p => p.FDR < 0.05).length}</p>
              <p className="text-xs text-muted-foreground mt-1">FDR &lt; 0.05</p>
            </div>
            <Target className="w-8 h-8 text-protein" />
          </div>
        </Card>
      </div>

      <Tabs defaultValue="degs" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="degs">DEGs</TabsTrigger>
          <TabsTrigger value="pathways">Pathways</TabsTrigger>
          <TabsTrigger value="phenotypes">Phenotypes</TabsTrigger>
          <TabsTrigger value="hypotheses">Hypotheses</TabsTrigger>
        </TabsList>

        {/* DEGs Tab */}
        <TabsContent value="degs" className="space-y-6">
          {completedNodes.includes("rna") && (
            <>
              {/* Volcano Plot */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Dna className="w-5 h-5 text-rna" />
                  Differential Expression - Volcano Plot
                </h3>
                <ChartContainer config={{}} className="h-[400px]">
                  <ScatterChart data={volcanoData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="x" 
                      name="Log2 Fold Change"
                      label={{ value: "Log2 Fold Change", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis 
                      dataKey="y" 
                      name="-Log10(p-value)"
                      label={{ value: "-Log10(p-value)", angle: -90, position: "insideLeft" }}
                    />
                    <ChartTooltip 
                      cursor={{ strokeDasharray: "3 3" }}
                      content={({ active, payload }) => {
                        if (active && payload && payload[0]) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                              <p className="font-semibold">{data.gene}</p>
                              <p className="text-sm">Log2FC: {data.log2fc.toFixed(2)}</p>
                              <p className="text-sm">p-value: {data.pval.toFixed(4)}</p>
                              <p className="text-sm">Direction: {data.direction}</p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Scatter 
                      dataKey="y" 
                      fill="#f59e0b"
                      shape={(props: any) => {
                        const { cx, cy, payload } = props;
                        const color = payload.significant 
                          ? (payload.direction === "up" ? "#10b981" : "#ef4444")
                          : "#94a3b8";
                        return <circle cx={cx} cy={cy} r={payload.significant ? 6 : 4} fill={color} />;
                      }}
                    />
                  </ScatterChart>
                </ChartContainer>
                <div className="flex gap-4 mt-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-600" />
                    <span>Upregulated (significant)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-600" />
                    <span>Downregulated (significant)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-slate-400" />
                    <span>Not significant</span>
                  </div>
                </div>
              </Card>

              {/* DEGs Table */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Top Differential Genes</h3>
                <ScrollArea className="h-[400px]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-secondary">
                      <tr>
                        <th className="text-left p-2 font-semibold">Gene</th>
                        <th className="text-right p-2 font-semibold">Log2FC</th>
                        <th className="text-right p-2 font-semibold">p-value</th>
                        <th className="text-right p-2 font-semibold">p-adj</th>
                        <th className="text-center p-2 font-semibold">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {degs
                        .sort((a, b) => Math.abs(b.log2fc) - Math.abs(a.log2fc))
                        .map((deg, i) => (
                          <tr key={i} className="border-b border-border hover:bg-secondary/50">
                            <td className="p-2 font-medium">{deg.gene}</td>
                            <td className="p-2 text-right">{deg.log2fc.toFixed(2)}</td>
                            <td className="p-2 text-right">{deg.pval.toFixed(4)}</td>
                            <td className="p-2 text-right">{deg.padj.toFixed(4)}</td>
                            <td className="p-2 text-center">
                              <Badge variant={deg.direction === "up" ? "default" : "destructive"}>
                                {deg.direction === "up" ? "↑ Up" : "↓ Down"}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </ScrollArea>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Pathways Tab */}
        <TabsContent value="pathways" className="space-y-6">
          {completedNodes.includes("rna") && (
            <>
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-protein" />
                  Pathway Enrichment Analysis
                </h3>
                <ChartContainer config={{}} className="h-[400px]">
                  <BarChart data={pathways.sort((a, b) => Math.abs(b.NES) - Math.abs(a.NES))} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" label={{ value: "Normalized Enrichment Score (NES)", position: "insideBottom", offset: -5 }} />
                    <YAxis dataKey="name" type="category" width={200} />
                    <ChartTooltip 
                      content={({ active, payload }) => {
                        if (active && payload && payload[0]) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                              <p className="font-semibold">{data.name}</p>
                              <p className="text-sm">NES: {data.NES.toFixed(2)}</p>
                              <p className="text-sm">FDR: {data.FDR.toFixed(4)}</p>
                              <p className="text-sm">Source: {data.source}</p>
                              <p className="text-sm">Genes: {data.genes}</p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar 
                      dataKey="NES" 
                      fill="hsl(var(--protein-pink))"
                      shape={(props: any) => {
                        const { payload, x, y, width, height } = props;
                        const color = payload.FDR < 0.05 ? (payload.NES > 0 ? "#10b981" : "#ef4444") : "#94a3b8";
                        return <rect x={x} y={y} width={width} height={height} fill={color} />;
                      }}
                    />
                  </BarChart>
                </ChartContainer>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {pathways
                  .filter(p => p.FDR < 0.05)
                  .sort((a, b) => Math.abs(b.NES) - Math.abs(a.NES))
                  .map((pathway, i) => (
                    <Card key={i} className="p-4">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-sm">{pathway.name}</h4>
                        <Badge variant={pathway.NES > 0 ? "default" : "secondary"}>
                          {pathway.source}
                        </Badge>
                      </div>
                      <div className="space-y-1 text-xs text-muted-foreground">
                        <p>NES: {pathway.NES.toFixed(2)}</p>
                        <p>FDR: {pathway.FDR.toFixed(4)}</p>
                        <p>Genes: {pathway.genes}</p>
                        <div className="mt-2">
                          <p className="text-xs font-medium mb-1">Key genes:</p>
                          <div className="flex flex-wrap gap-1">
                            {pathway.member_genes.slice(0, 5).map((gene, j) => (
                              <Badge key={j} variant="outline" className="text-xs">
                                {gene}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </Card>
                  ))}
              </div>
            </>
          )}
        </TabsContent>

        {/* Phenotypes Tab */}
        <TabsContent value="phenotypes" className="space-y-6">
          {completedNodes.includes("protein") && (
            <>
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-protein" />
                  Predicted Phenotypes
                </h3>
                <ChartContainer config={{}} className="h-[400px]">
                  <BarChart data={phenotypes.sort((a, b) => Math.abs(b.score) - Math.abs(a.score))} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[-1, 1]} label={{ value: "Phenotype Score", position: "insideBottom", offset: -5 }} />
                    <YAxis dataKey="name" type="category" width={180} />
                    <ChartTooltip 
                      content={({ active, payload }) => {
                        if (active && payload && payload[0]) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                              <p className="font-semibold">{data.name}</p>
                              <p className="text-sm">Score: {data.score.toFixed(2)}</p>
                              <p className="text-sm">Direction: {data.direction}</p>
                              <p className="text-sm">Confidence: {(data.confidence * 100).toFixed(0)}%</p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar 
                      dataKey="score"
                      fill="hsl(var(--protein-pink))"
                      shape={(props: any) => {
                        const { payload, x, y, width, height } = props;
                        const color = payload.score > 0 ? "#10b981" : "#ef4444";
                        return <rect x={x} y={y} width={width} height={height} fill={color} />;
                      }}
                    />
                  </BarChart>
                </ChartContainer>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {phenotypes.map((pheno, i) => (
                  <Card key={i} className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-semibold text-sm">{pheno.name}</h4>
                      <Badge variant={pheno.score > 0.6 ? "default" : "secondary"}>
                        {pheno.direction}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Score:</span>
                        <span className="font-semibold">{pheno.score.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Confidence:</span>
                        <span className="font-semibold">{(pheno.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </>
          )}
        </TabsContent>

        {/* Hypotheses Tab */}
        <TabsContent value="hypotheses" className="space-y-6">
          {completedNodes.includes("protein") && (
            <>
              {hypotheses.map((hyp, i) => (
                <Card key={i} className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Lightbulb className="w-5 h-5 text-yellow-500" />
                      <h3 className="text-lg font-semibold">{hyp.id}: {hyp.statement}</h3>
                    </div>
                    <Badge variant={hyp.literature_support.support_level === "strong" ? "default" : "secondary"}>
                      {hyp.literature_support.support_level} support
                    </Badge>
                  </div>

                  <div className="space-y-4">
                    {/* Mechanism */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-muted-foreground">Mechanism</h4>
                      <ul className="list-disc list-inside space-y-1 text-sm">
                        {hyp.mechanism.map((step, j) => (
                          <li key={j} className="text-muted-foreground">{step}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Phenotype Support */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-muted-foreground">Phenotype Support</h4>
                      <div className="bg-secondary/50 rounded-lg p-3 text-sm">
                        <p className="font-medium">{hyp.phenotype_support.primary_phenotype}</p>
                        <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
                          <div>
                            <span className="text-muted-foreground">Score: </span>
                            <span className="font-semibold">{hyp.phenotype_support.score.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Direction: </span>
                            <span className="font-semibold">{hyp.phenotype_support.direction}</span>
                          </div>
                        </div>
                        <div className="mt-2">
                          <p className="text-xs text-muted-foreground mb-1">Supporting genes:</p>
                          <div className="flex flex-wrap gap-1">
                            {hyp.phenotype_support.supporting_genes.map((gene, j) => (
                              <Badge key={j} variant="outline" className="text-xs">
                                {gene}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Predicted Readouts */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-muted-foreground">Predicted Readouts</h4>
                      <div className="flex flex-wrap gap-2">
                        {hyp.predicted_readouts.map((readout, j) => (
                          <Badge key={j} variant="outline" className="text-xs">
                            {readout}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Literature Support */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-muted-foreground">Literature Support</h4>
                      <div className="flex items-center gap-2 text-sm">
                        <BookOpen className="w-4 h-4 text-muted-foreground" />
                        <span>{hyp.literature_support.papers} papers found</span>
                        {hyp.literature_support.pmids.length > 0 && (
                          <span className="text-xs text-muted-foreground">
                            ({hyp.literature_support.pmids.slice(0, 2).join(", ")})
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Suggested Experiments */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-muted-foreground">Suggested Experiments</h4>
                      <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                        {hyp.experiments.map((exp, j) => (
                          <li key={j}>{exp}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </Card>
              ))}
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};
