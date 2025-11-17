/**
 * Extract perturbation information from user queries
 * Mimics the logic from llm/input.py
 */

export interface PerturbationInfo {
  target: string | null;
  target2?: string | null;
  type: "KO" | "KD" | "OE" | "drug" | "unknown";
  type2?: "KO" | "KD" | "OE" | "drug" | "unknown";
  confidence: number;
  isComparison: boolean;
  originalQuery: string;
}

// Common gene names (sample - in real app, load from STATE model)
const COMMON_GENES = [
  "TP53", "BRCA1", "BRCA2", "EGFR", "MYC", "CDKN2A", "PIK3CA", "KRAS",
  "JAK1", "JAK2", "STAT3", "AKT1", "PTEN", "RB1", "MDM2", "BAX", "BCL2",
  "CDKN1A", "CCND1", "CASP3", "FAS", "GADD45A", "ATM", "CHEK2", "ACTB", "AARS"
];

// Common drug names (sample - in real app, load from ST-Tahoe model)
const COMMON_DRUGS = [
  "imatinib", "aspirin", "verapamil", "crizotinib", "doxorubicin", "paclitaxel",
  "cisplatin", "methotrexate", "5-fluorouracil", "tamoxifen"
];

export function extractPerturbationInfo(query: string): PerturbationInfo {
  const lowerQuery = query.toLowerCase();
  
  // Detect perturbation type keywords
  const hasKO = /\b(knockout|ko|knock out|delete|deletion)\b/i.test(query);
  const hasKD = /\b(knockdown|kd|knock down|silence|silencing|inhibit|inhibition)\b/i.test(query);
  const hasOE = /\b(overexpress|oe|over express|overexpression|upregulate|upregulation)\b/i.test(query);
  const hasDrug = /\b(drug|compound|treatment|therapeutic|medication)\b/i.test(query);
  
  // Extract gene names (uppercase, 2-10 chars, alphanumeric)
  const genePattern = /\b([A-Z]{2,10}[0-9A-Z]*)\b/g;
  const geneMatches = query.match(genePattern) || [];
  
  // Extract drug names (lowercase, longer words, or quoted strings)
  const drugPattern = /\b([a-z]{4,}(?:[a-z]+)?)\b/g;
  const drugMatches = lowerQuery.match(drugPattern) || [];
  
  // Check if it's a comparison
  const isComparison = /\b(vs|versus|compare|comparison|between|and)\b/i.test(query);
  
  // Determine perturbation type
  let type: "KO" | "KD" | "OE" | "drug" | "unknown" = "unknown";
  if (hasKO) type = "KO";
  else if (hasKD) type = "KD";
  else if (hasOE) type = "OE";
  else if (hasDrug) type = "drug";
  
  // Find target
  let target: string | null = null;
  let target2: string | null = null;
  
  if (type === "drug" || hasDrug) {
    // Look for drug names
    const foundDrug = drugMatches.find(d => COMMON_DRUGS.includes(d));
    if (foundDrug) {
      target = foundDrug;
    } else if (drugMatches.length > 0) {
      target = drugMatches[0];
    }
  } else {
    // Look for gene names
    const foundGene = geneMatches.find(g => COMMON_GENES.includes(g));
    if (foundGene) {
      target = foundGene;
    } else if (geneMatches.length > 0) {
      target = geneMatches[0];
    }
  }
  
  // For comparisons, try to find second target
  if (isComparison && target) {
    const remainingText = query.replace(new RegExp(target, "gi"), "");
    const remainingGenes = remainingText.match(genePattern) || [];
    const remainingDrugs = remainingText.toLowerCase().match(drugPattern) || [];
    
    if (type === "drug") {
      const secondDrug = remainingDrugs.find(d => COMMON_DRUGS.includes(d));
      if (secondDrug) target2 = secondDrug;
    } else {
      const secondGene = remainingGenes.find(g => COMMON_GENES.includes(g));
      if (secondGene) target2 = secondGene;
    }
  }
  
  // Calculate confidence
  let confidence = 0.5;
  if (target && type !== "unknown") confidence = 0.8;
  if (target && COMMON_GENES.includes(target) || COMMON_DRUGS.includes(target?.toLowerCase() || "")) {
    confidence = 0.9;
  }
  if (hasKO || hasKD || hasOE) confidence = Math.min(confidence + 0.1, 1.0);
  
  return {
    target,
    target2: target2 || undefined,
    type,
    isComparison,
    confidence,
    originalQuery: query,
  };
}

