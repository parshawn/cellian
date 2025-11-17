"""Hypothesis generation agent with literature support."""

from typing import Dict, List, Any, Optional
from .futurehouse_client import search_literature, get_pmids, format_citation


def generate_hypotheses(
    payload: Dict[str, Any],
    save_preliminary_report: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrate hypothesis generation:
    - Select top phenotypes and pathways
    - Draft mechanistic hypotheses
    - Retrieve literature via Edison Scientific (PaperQA)
    - Classify support
    - Generate report-friendly JSON
    
    Args:
        payload: Input payload with keys:
            - context: dict with perturbation, cell_type, species, user_question
            - validated_edges: list of edge dicts (source, target, direction, confidence)
            - deg_list: list of DEG dicts (gene, log2fc, pval)
            - pathways: list of pathway dicts (id, name, NES, FDR, member_genes)
            - phenotypes: list of phenotype dicts (phenotype_id, name, score, direction, ...)
            - evidence: dict with datasets, papers (optional)
        save_preliminary_report: Optional dict with keys:
            - results: Results dict with perturbations and comparison
            - query: User query string
            - context: Context dict
            If provided, will save preliminary report immediately after LLM generation, before literature search
    
    Returns:
        Dictionary with hypotheses:
        {
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": "...",
                    "mechanism": [...],
                    "phenotype_support": {...},
                    "predicted_readouts": [...],
                    "literature_support": {...},
                    "experiments": [...],
                    "speculation_notes": "..."
                },
                ...
            ]
        }
    """
    context = payload.get("context", {})
    validated_edges = payload.get("validated_edges", [])
    deg_list = payload.get("deg_list", [])
    pathways = payload.get("pathways", [])
    phenotypes = payload.get("phenotypes", [])
    evidence = payload.get("evidence", {})
    
    perturbation = context.get("perturbation", "Unknown")
    cell_type = context.get("cell_type", "")
    species = context.get("species", "")
    user_question = context.get("user_question", "")
    perturbation_type = context.get("perturbation_type", "")
    perturbation1 = context.get("perturbation1", "")
    perturbation2 = context.get("perturbation2", "")
    
    # Identify top phenotypes (score >= 0.6)
    top_phenotypes = [
        p for p in phenotypes
        if p.get("score", 0.0) >= 0.6
    ]
    top_phenotypes.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Identify top pathways (p-value <= 0.05, not FDR)
    # Use p-value field (can be pval, pvalue, or P-value depending on source)
    top_pathways = []
    for p in pathways:
        pval = p.get("pval") or p.get("pvalue") or p.get("P-value") or p.get("P_value") or 1.0
        if isinstance(pval, (int, float)) and pval <= 0.05:
            top_pathways.append(p)
    top_pathways.sort(key=lambda x: abs(x.get("NES", 0.0)), reverse=True)
    
    # Step 1: Use LLM to generate hypotheses based on ALL combined results
    # First, generate hypotheses using LLM analysis of all data
    print(f"[LLM Hypothesis Generation] Analyzing all combined results...")
    print(f"  - DEGs: {len(deg_list)} genes")
    print(f"  - Pathways: {len(top_pathways)} enriched pathways (p ≤ 0.05)")
    print(f"  - Validated edges: {len(validated_edges)} regulatory edges")
    print(f"  Generating mechanistic hypotheses using LLM...")
    print(f"  (This may take 30-60 seconds for LLM analysis...)\n")
    import sys
    sys.stdout.flush()
    
    llm_hypotheses = _generate_llm_hypotheses(
        perturbation=perturbation,
        deg_list=deg_list,
        pathways=top_pathways,
        validated_edges=validated_edges,
        cell_type=cell_type,
        species=species,
        context=context,
        user_question=user_question,
        perturbation_type=perturbation_type,
        perturbation1=perturbation1,
        perturbation2=perturbation2
    )
    
    print(f"[LLM Hypothesis Generation] LLM analysis completed\n")
    sys.stdout.flush()
    
    # Display generated hypotheses before literature search
    if llm_hypotheses:
        print(f"[LLM Hypothesis Generation] ✓ Generated {len(llm_hypotheses)} hypotheses:")
        for i, hyp in enumerate(llm_hypotheses, 1):
            statement = hyp.get("statement", "")
            mechanism = hyp.get("mechanism", [])
            key_pathways = hyp.get("key_pathways", [])
            key_genes = hyp.get("key_genes", [])
            
            print(f"\n  Hypothesis {i}:")
            print(f"    Statement: {statement}")
            if mechanism:
                print(f"    Mechanism: {len(mechanism)} steps")
                # Show mechanism steps
                for j, step in enumerate(mechanism[:3], 1):
                    print(f"      {j}. {step[:100]}...")
                if len(mechanism) > 3:
                    print(f"      ... ({len(mechanism) - 3} more steps)")
            if key_pathways:
                print(f"    Key pathways: {', '.join(key_pathways[:3])}")
            if key_genes:
                print(f"    Key genes: {', '.join(key_genes[:3])}")
        print()
        print(f"[LLM Hypothesis Generation] ✓ Generated {len(llm_hypotheses)} hypotheses")
        print(f"    Full details: statement, mechanism ({sum(len(h.get('mechanism', [])) for h in llm_hypotheses)} total steps),")
        print(f"                 key pathways ({sum(len(h.get('key_pathways', [])) for h in llm_hypotheses)} total),")
        print(f"                 key genes ({sum(len(h.get('key_genes', [])) for h in llm_hypotheses)} total)")
        print(f"    Note: These will be saved to preliminary_report.md (without literature search results)")
        print(f"    Proceeding to literature search...\n")
    import sys
    sys.stdout.flush()
    
    # Store LLM-generated hypotheses (before literature enrichment) for preliminary report
    # This allows the preliminary report to show hypotheses before literature search completes
    llm_hypotheses_for_report = []
    if llm_hypotheses:
        for hyp in llm_hypotheses[:5]:
            llm_hypotheses_for_report.append({
                "statement": hyp.get("statement", ""),
                "mechanism": hyp.get("mechanism", []),
                "key_pathways": hyp.get("key_pathways", []),
                "key_genes": hyp.get("key_genes", []),
                "literature_status": "pending"  # Will be added in final report
            })
    
    # Save preliminary report IMMEDIATELY after LLM generation, BEFORE literature search
    if save_preliminary_report and llm_hypotheses_for_report:
        try:
            from .perturbation_orchestrator import _generate_preliminary_report
            from pathlib import Path
            import sys
            
            print(f"\n[Preliminary Report] Saving preliminary report (before literature search)...")
            sys.stdout.flush()
            
            preliminary_report = _generate_preliminary_report(
                results=save_preliminary_report.get("results", {}),
                query=save_preliminary_report.get("query", ""),
                context=save_preliminary_report.get("context", {}),
                llm_hypotheses=llm_hypotheses_for_report
            )
            
            if preliminary_report:
                main_output_dir = Path(__file__).parent / "perturbation_outputs"
                main_output_dir.mkdir(parents=True, exist_ok=True)
                prelim_report_path = main_output_dir / "preliminary_report.md"
                with open(prelim_report_path, 'w') as f:
                    f.write(preliminary_report)
                
                # Store the path in results so orchestrator knows it was saved
                save_preliminary_report.get("results", {})["preliminary_report"] = str(prelim_report_path)
                
                print(f"  ✓ Saved preliminary report: {prelim_report_path}")
                print(f"    Length: {len(preliminary_report)} characters")
                print(f"    Contains: All perturbations, comparison, {len(llm_hypotheses_for_report)} LLM-generated hypotheses (full details)")
                print(f"    Note: Literature search will now run... Results will be added to final comprehensive report\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"  ⚠️  Could not save preliminary report: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    # Step 2: Perform literature search for all generated hypotheses
    # If LLM generated hypotheses, use them; otherwise fall back to template-based
    hypotheses = []
    hypothesis_counter = 1
    
    if llm_hypotheses:
        print(f"[Literature Search] Starting literature search for {len(llm_hypotheses)} hypotheses...")
        print(f"  This may take a few moments (Edison Scientific API calls)...")
        print(f"  Searching papers for each hypothesis...\n")
        sys.stdout.flush()
        
        # Use LLM-generated hypotheses and enrich with literature search
        for i, llm_hyp in enumerate(llm_hypotheses, 1):
            if hypothesis_counter > 5:  # Limit to 5 hypotheses
                print(f"  [Skipping remaining hypotheses - limit of 5 reached]")
                sys.stdout.flush()
                break
            
            statement = llm_hyp.get("statement", "")
            print(f"  [{i}/{min(len(llm_hypotheses), 5)}] Searching literature for hypothesis {i}...")
            print(f"      Statement: {statement[:100]}...")
            sys.stdout.flush()
            
            # For each LLM hypothesis, search literature and add support
            enriched_hyp = _enrich_hypothesis_with_literature(
                hypothesis_dict=llm_hyp,
                hypothesis_id=f"H{hypothesis_counter}",
                perturbation=perturbation,
                cell_type=cell_type,
                validated_edges=validated_edges
            )
            if enriched_hyp:
                hypotheses.append(enriched_hyp)
                # Show how many papers were found
                lit_support = enriched_hyp.get("literature_support", {})
                num_papers = len(lit_support.get("supporting_papers_full", []))
                overall = lit_support.get("overall", "unknown")
                print(f"      ✓ Found {num_papers} papers (support level: {overall})")
                sys.stdout.flush()
                hypothesis_counter += 1
            else:
                print(f"      ⚠️  Could not enrich hypothesis with literature")
                sys.stdout.flush()
        
        print(f"\n[Literature Search] ✓ Completed literature search for all hypotheses")
        print(f"    Total hypotheses with literature: {len(hypotheses)}")
        print(f"    Literature results will be added to final comprehensive report\n")
        sys.stdout.flush()
    
    # Fall back to template-based if LLM didn't generate hypotheses or we need more
    if len(hypotheses) < 3:  # If we have fewer than 3 LLM hypotheses, add template-based ones
        # Hypothesis 1: Primary phenotype-driven hypothesis (only if phenotypes available)
        if top_phenotypes:
            primary_phenotype = top_phenotypes[0]
            hyp = _create_phenotype_hypothesis(
                hypothesis_id=f"H{hypothesis_counter}",
                phenotype=primary_phenotype,
                pathways=top_pathways,
                validated_edges=validated_edges,
                deg_list=deg_list,
                perturbation=perturbation,
                cell_type=cell_type,
                species=species
            )
            if hyp:
                hypotheses.append(hyp)
                hypothesis_counter += 1
        
        # Hypothesis 2-3: Top pathway-driven hypotheses
        for pathway in top_pathways[:2]:
            if hypothesis_counter > 5:
                break
            
            # Check if this pathway supports a phenotype
            pathway_phenotypes = [
                p for p in top_phenotypes
                if any(
                    path.get("id") == pathway.get("id")
                    for path in p.get("supporting_pathways", [])
                )
            ]
            
            if pathway_phenotypes or len(pathways) > 0:
                hyp = _create_pathway_hypothesis(
                    hypothesis_id=f"H{hypothesis_counter}",
                    pathway=pathway,
                    phenotypes=pathway_phenotypes or top_phenotypes[:1],
                    validated_edges=validated_edges,
                    deg_list=deg_list,
                    perturbation=perturbation,
                    cell_type=cell_type,
                    species=species
                )
                if hyp:
                    hypotheses.append(hyp)
                    hypothesis_counter += 1
        
        # Hypothesis 4-5: Edge-driven hypotheses (if we have strong edges)
        strong_edges = [
            e for e in validated_edges
            if e.get("confidence", 0.0) >= 0.7
        ]
        
        for edge in strong_edges[:2]:
            if hypothesis_counter > 5:
                break
            
            # Find related pathways and phenotypes
            source_gene = edge.get("source", "")
            target_gene = edge.get("target", "")
            
            related_pathways = [
                p for p in top_pathways
                if source_gene in p.get("member_genes", []) or target_gene in p.get("member_genes", [])
            ]
            
            related_phenotypes = [
                p for p in top_phenotypes
                if source_gene in p.get("supporting_genes", []) or target_gene in p.get("supporting_genes", [])
            ]
            
            if related_pathways or related_phenotypes:
                hyp = _create_edge_hypothesis(
                    hypothesis_id=f"H{hypothesis_counter}",
                    edge=edge,
                    pathways=related_pathways,
                    phenotypes=related_phenotypes or top_phenotypes[:1],
                    deg_list=deg_list,
                    perturbation=perturbation,
                    cell_type=cell_type,
                    species=species
                )
                if hyp:
                    hypotheses.append(hyp)
                    hypothesis_counter += 1
    
    # If no hypotheses generated, provide LLM-powered explanation and perform literature search
    if not hypotheses:
        # Perform literature search anyway to gather evidence
        query_parts = [perturbation]
        if cell_type:
            query_parts.append(cell_type)
        if top_pathways:
            query_parts.append(top_pathways[0].get("name", ""))
        elif deg_list:
            # Use top DEGs
            sorted_degs = sorted(deg_list, key=lambda x: abs(x.get("log2fc", 0.0)), reverse=True)
            query_parts.extend([d.get("gene", "") for d in sorted_degs[:3]])
        
        query = " ".join([q for q in query_parts if q])
        literature_papers = search_literature(query, k=10) if query else []
        
        # Generate explanation using LLM
        explanation = _generate_no_hypotheses_explanation(
            perturbation=perturbation,
            top_phenotypes=top_phenotypes,
            top_pathways=top_pathways,
            deg_list=deg_list,
            pathways=pathways,
            phenotypes=phenotypes,
            literature_papers=literature_papers
        )
        
        return {
            "hypotheses": [],
            "summary": {
                "n_hypotheses": 0,
                "top_phenotype": top_phenotypes[0].get("name") if top_phenotypes else None,
                "top_pathway": top_pathways[0].get("name") if top_pathways else None
            },
            "no_hypotheses_explanation": explanation,
            "literature_search_performed": True,
            "literature_papers_found": len(literature_papers),
            "supporting_papers_full": literature_papers[:5] if literature_papers else []
        }
    
    result = {
        "hypotheses": hypotheses,
        "summary": {
            "n_hypotheses": len(hypotheses),
            "top_phenotype": top_phenotypes[0].get("name") if top_phenotypes else None,
            "top_pathway": top_pathways[0].get("name") if top_pathways else None
        }
    }
    
    # Store LLM-generated hypotheses (before literature enrichment) for preliminary report
    # This allows showing hypotheses immediately before literature search completes
    if llm_hypotheses_for_report:
        result["llm_generated_hypotheses_before_literature"] = llm_hypotheses_for_report
    
    return result


def _create_phenotype_hypothesis(
    hypothesis_id: str,
    phenotype: Dict[str, Any],
    pathways: List[Dict[str, Any]],
    validated_edges: List[Dict[str, Any]],
    deg_list: List[Dict[str, Any]],
    perturbation: str,
    cell_type: str,
    species: str
) -> Optional[Dict[str, Any]]:
    """Create a phenotype-driven hypothesis."""
    phenotype_name = phenotype.get("name", "Unknown")
    phenotype_id = phenotype.get("phenotype_id", "")
    direction = phenotype.get("direction", "mixed")
    supporting_genes = phenotype.get("supporting_genes", [])
    supporting_pathways = phenotype.get("supporting_pathways", [])
    
    # Build mechanism chain
    mechanism = [f"Perturbation: {perturbation}"]
    
    # Add key genes
    if supporting_genes:
        top_genes = supporting_genes[:3]
        mechanism.append(f"Affects genes: {', '.join(top_genes)}")
    
    # Add pathways
    if supporting_pathways:
        top_pathway = supporting_pathways[0]
        mechanism.append(f"Enriches pathway: {top_pathway.get('name', 'Unknown')}")
    
    # Final phenotype
    mechanism.append(f"Predicts phenotype: {phenotype_name} ({direction})")
    
    # Draft statement
    direction_text = "increases" if direction == "increase" else "decreases" if direction == "decrease" else "modulates"
    statement = f"{perturbation} {direction_text} {phenotype_name.lower()}"
    if cell_type:
        statement += f" in {cell_type} cells"
    
    # Build query for literature search
    query_parts = [perturbation, phenotype_name]
    if cell_type:
        query_parts.append(cell_type)
    if supporting_genes:
        query_parts.extend(supporting_genes[:2])
    
    query = " ".join(query_parts)
    
    # Search literature
    literature_papers = search_literature(query, k=10)
    
    # Classify literature support
    literature_support = _classify_literature_support(
        papers=literature_papers,
        phenotype=phenotype_name,
        perturbation=perturbation,
        direction=direction
    )
    
    # Generate predicted readouts
    predicted_readouts = _generate_predicted_readouts(
        phenotype=phenotype,
        pathways=supporting_pathways
    )
    
    # Suggest experiments
    experiments = _suggest_experiments(
        phenotype=phenotype,
        perturbation=perturbation,
        cell_type=cell_type
    )
    
    # Speculation notes
    speculation_notes = _generate_speculation_notes(
        phenotype=phenotype,
        validated_edges=validated_edges,
        literature_papers=literature_papers
    )
    
    return {
        "id": hypothesis_id,
        "statement": statement,
        "mechanism": mechanism,
        "phenotype_support": {
            "primary_phenotype": phenotype_name,
            "phenotype_id": phenotype_id,
            "score": phenotype.get("score", 0.0),
            "direction": direction,
            "supporting_genes": supporting_genes[:10],  # Top 10
            "supporting_pathways": [p.get("name") for p in supporting_pathways[:5]]
        },
        "predicted_readouts": predicted_readouts,
        "literature_support": literature_support,
        "experiments": experiments,
        "speculation_notes": speculation_notes
    }


def _create_pathway_hypothesis(
    hypothesis_id: str,
    pathway: Dict[str, Any],
    phenotypes: List[Dict[str, Any]],
    validated_edges: List[Dict[str, Any]],
    deg_list: List[Dict[str, Any]],
    perturbation: str,
    cell_type: str,
    species: str
) -> Optional[Dict[str, Any]]:
    """Create a pathway-driven hypothesis."""
    pathway_name = pathway.get("name", "Unknown")
    pathway_id = pathway.get("id", "")
    member_genes = pathway.get("member_genes", [])
    nes = pathway.get("NES", 0.0)
    fdr = pathway.get("FDR", 1.0)
    
    # Find associated phenotype
    primary_phenotype = phenotypes[0] if phenotypes else None
    
    # Build mechanism
    mechanism = [f"Perturbation: {perturbation}"]
    mechanism.append(f"Enriches pathway: {pathway_name} (NES={nes:.2f}, FDR={fdr:.3f})")
    
    if member_genes:
        mechanism.append(f"Affects pathway genes: {', '.join(member_genes[:5])}")
    
    if primary_phenotype:
        mechanism.append(f"Predicts phenotype: {primary_phenotype.get('name', 'Unknown')}")
    
    # Draft statement
    pathway_direction = "activates" if nes > 0 else "inhibits"
    statement = f"{perturbation} {pathway_direction} {pathway_name}"
    if primary_phenotype:
        statement += f", leading to {primary_phenotype.get('name', 'phenotypic changes').lower()}"
    
    # Literature search
    query = f"{perturbation} {pathway_name}"
    if cell_type:
        query += f" {cell_type}"
    
    literature_papers = search_literature(query, k=10)
    
    # Classify support
    literature_support = _classify_literature_support(
        papers=literature_papers,
        pathway=pathway_name,
        perturbation=perturbation
    )
    
    # Predicted readouts
    predicted_readouts = _generate_predicted_readouts(
        pathway=pathway,
        phenotypes=phenotypes
    )
    
    # Experiments
    experiments = _suggest_experiments(
        pathway=pathway,
        perturbation=perturbation,
        cell_type=cell_type
    )
    
    # Speculation
    speculation_notes = _generate_speculation_notes(
        pathway=pathway,
        validated_edges=validated_edges,
        literature_papers=literature_papers
    )
    
    return {
        "id": hypothesis_id,
        "statement": statement,
        "mechanism": mechanism,
        "phenotype_support": {
            "primary_phenotype": primary_phenotype.get("name") if primary_phenotype else None,
            "phenotype_id": primary_phenotype.get("phenotype_id") if primary_phenotype else None,
            "score": primary_phenotype.get("score") if primary_phenotype else None,
            "direction": primary_phenotype.get("direction") if primary_phenotype else "unknown",
            "supporting_pathways": [pathway_name]
        } if primary_phenotype else None,
        "predicted_readouts": predicted_readouts,
        "literature_support": literature_support,
        "experiments": experiments,
        "speculation_notes": speculation_notes
    }


def _create_edge_hypothesis(
    hypothesis_id: str,
    edge: Dict[str, Any],
    pathways: List[Dict[str, Any]],
    phenotypes: List[Dict[str, Any]],
    deg_list: List[Dict[str, Any]],
    perturbation: str,
    cell_type: str,
    species: str
) -> Optional[Dict[str, Any]]:
    """Create an edge-driven hypothesis."""
    source = edge.get("source", "")
    target = edge.get("target", "")
    direction = edge.get("direction", "unknown")
    confidence = edge.get("confidence", 0.0)
    
    primary_phenotype = phenotypes[0] if phenotypes else None
    
    # Build mechanism
    mechanism = [f"Perturbation: {perturbation}"]
    mechanism.append(f"Directly affects {source} → {target} ({direction}, confidence={confidence:.2f})")
    
    if pathways:
        mechanism.append(f"Propagates through pathway: {pathways[0].get('name', 'Unknown')}")
    
    if primary_phenotype:
        mechanism.append(f"Predicts phenotype: {primary_phenotype.get('name', 'Unknown')}")
    
    # Statement
    direction_text = "upregulates" if direction == "up" else "downregulates" if direction == "down" else "affects"
    statement = f"{perturbation} {direction_text} {target} via {source}, leading to {primary_phenotype.get('name', 'phenotypic changes').lower() if primary_phenotype else 'cellular changes'}"
    
    # Literature search
    query = f"{source} {target} {direction} {perturbation}"
    if cell_type:
        query += f" {cell_type}"
    
    literature_papers = search_literature(query, k=10)
    
    # Classify support
    literature_support = _classify_literature_support(
        papers=literature_papers,
        edge=edge,
        perturbation=perturbation
    )
    
    # Predicted readouts
    predicted_readouts = _generate_predicted_readouts(
        edge=edge,
        pathways=pathways,
        phenotypes=phenotypes
    )
    
    # Experiments
    experiments = _suggest_experiments(
        edge=edge,
        perturbation=perturbation,
        cell_type=cell_type
    )
    
    # Speculation
    speculation_notes = _generate_speculation_notes(
        edge=edge,
        validated_edges=[edge],
        literature_papers=literature_papers
    )
    
    return {
        "id": hypothesis_id,
        "statement": statement,
        "mechanism": mechanism,
        "phenotype_support": {
            "primary_phenotype": primary_phenotype.get("name") if primary_phenotype else None,
            "phenotype_id": primary_phenotype.get("phenotype_id") if primary_phenotype else None,
            "score": primary_phenotype.get("score") if primary_phenotype else None,
            "direction": primary_phenotype.get("direction") if primary_phenotype else "unknown"
        } if primary_phenotype else None,
        "predicted_readouts": predicted_readouts,
        "literature_support": literature_support,
        "experiments": experiments,
        "speculation_notes": speculation_notes
    }


def _classify_literature_support(
    papers: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Classify literature support level."""
    if not papers or len(papers) < 3:
        return {
            "overall": "weak",
            "supporting_papers": get_pmids(papers),
            "supporting_papers_full": papers[:5],  # Store full paper data for citation formatting
            "contradicting_papers": [],
            "summary": "Limited literature found. Hypothesis requires experimental validation."
        }
    
    # Simple heuristic: count high-scoring papers
    high_score_papers = [p for p in papers if p.get("score", 0.0) >= 0.7]
    
    if len(high_score_papers) >= 5:
        overall = "strong"
        summary = f"Strong literature support found ({len(high_score_papers)} highly relevant papers)."
    elif len(high_score_papers) >= 2:
        overall = "mixed"
        summary = f"Mixed evidence found ({len(high_score_papers)} highly relevant papers, {len(papers)} total)."
    else:
        overall = "weak"
        summary = f"Weak or indirect evidence. Found {len(papers)} potentially relevant papers but limited direct support."
    
    # Store both PMIDs (for quick reference) and full paper data (for citation formatting)
    top_papers = papers[:5]
    return {
        "overall": overall,
        "supporting_papers": get_pmids(top_papers),  # Top 5 PMIDs for quick reference
        "supporting_papers_full": top_papers,  # Full paper data for formatted citations
        "contradicting_papers": [],  # Would require manual review or NLP
        "summary": summary
    }


def _generate_predicted_readouts(**kwargs) -> List[str]:
    """Generate predicted experimental readouts."""
    readouts = []
    
    phenotype = kwargs.get("phenotype")
    pathways = kwargs.get("pathways", [])
    edge = kwargs.get("edge")
    
    if phenotype:
        phenotype_name = phenotype.get("name", "")
        direction = phenotype.get("direction", "")
        
        if "apoptosis" in phenotype_name.lower():
            readouts.append("Caspase-3/7 activity (increase)" if direction == "increase" else "Caspase-3/7 activity (decrease)")
            readouts.append("Annexin V staining (flow cytometry)")
        
        if "proliferation" in phenotype_name.lower():
            readouts.append("Cell count (increase)" if direction == "increase" else "Cell count (decrease)")
            readouts.append("Ki-67 staining (immunofluorescence)")
        
        if "differentiation" in phenotype_name.lower():
            readouts.append("Lineage-specific marker expression (RNA-seq or IF)")
    
    if pathways:
        pathway = pathways[0] if pathways else {}
        pathway_name = pathway.get("name", "")
        
        if "apoptosis" in pathway_name.lower():
            readouts.append("TUNEL assay (DNA fragmentation)")
        
        if "cell cycle" in pathway_name.lower():
            readouts.append("Cell cycle analysis (PI staining + flow cytometry)")
    
    if edge:
        readouts.append(f"Expression of {edge.get('target', 'target gene')} (qRT-PCR or RNA-seq)")
    
    # Default readouts if none specified
    if not readouts:
        readouts.extend([
            "RNA-seq transcriptome profiling",
            "Protein expression (Western blot or mass spectrometry)",
            "Cell viability assay (MTT or CellTiter-Glo)"
        ])
    
    return readouts[:5]  # Top 5


def _suggest_experiments(**kwargs) -> List[str]:
    """Suggest concrete experiments."""
    experiments = []
    
    perturbation = kwargs.get("perturbation", "")
    cell_type = kwargs.get("cell_type", "")
    phenotype = kwargs.get("phenotype", {})
    pathway = kwargs.get("pathway", {})
    edge = kwargs.get("edge", {})
    
    base_experiment = f"Perform {perturbation} in {cell_type} cells"
    
    if phenotype:
        phenotype_name = phenotype.get("name", "")
        experiments.append(f"{base_experiment} and measure {phenotype_name.lower()} markers at 24h, 48h, and 72h post-perturbation")
        
        if "apoptosis" in phenotype_name.lower():
            experiments.append("Assay: Annexin V/PI staining + flow cytometry at 48h")
    
    if pathway:
        pathway_name = pathway.get("name", "")
        experiments.append(f"{base_experiment} and perform pathway activity assay for {pathway_name} at 24h")
    
    if edge:
        source = edge.get("source", "")
        target = edge.get("target", "")
        experiments.append(f"Validate {source} → {target} interaction using ChIP-seq or co-IP")
    
    # Default experiments
    if not experiments:
        experiments.append(f"{base_experiment} and perform transcriptome profiling (RNA-seq) at 24h")
        experiments.append("Validate top DEGs using qRT-PCR with independent replicates")
    
    return experiments[:4]  # Top 4


def _generate_speculation_notes(**kwargs) -> str:
    """Generate notes on what is inferred vs directly supported."""
    notes_parts = []
    
    validated_edges = kwargs.get("validated_edges", [])
    literature_papers = kwargs.get("literature_papers", [])
    phenotype = kwargs.get("phenotype", {})
    pathway = kwargs.get("pathway", {})
    
    # Check direct evidence
    if validated_edges:
        notes_parts.append(f"Directly supported by {len(validated_edges)} validated regulatory edges.")
    
    if literature_papers and len(literature_papers) >= 3:
        notes_parts.append(f"Literature support: {len(literature_papers)} relevant papers found.")
    else:
        notes_parts.append("Limited direct literature evidence - hypothesis requires experimental validation.")
    
    # Check inferred parts
    if phenotype:
        score = phenotype.get("score", 0.0)
        if score < 0.8:
            notes_parts.append(f"Phenotype prediction (score={score:.2f}) is moderate and inferred from pathway/gene signatures.")
    
    if pathway:
        fdr = pathway.get("FDR", 1.0)
        if fdr > 0.01:
            notes_parts.append(f"Pathway enrichment (FDR={fdr:.3f}) is suggestive but not highly significant.")
    
    if not validated_edges and not literature_papers:
        notes_parts.append("This hypothesis is largely inferred from computational predictions and requires strong experimental validation.")
    
    return " ".join(notes_parts) if notes_parts else "Hypothesis combines computational predictions with limited literature evidence."


def _generate_llm_hypotheses(
    perturbation: str,
    deg_list: List[Dict[str, Any]],
    pathways: List[Dict[str, Any]],
    validated_edges: List[Dict[str, Any]],
    cell_type: str,
    species: str,
    context: Dict[str, Any],
    user_question: str = "",
    perturbation_type: str = "",
    perturbation1: str = "",
    perturbation2: str = ""
) -> List[Dict[str, Any]]:
    """
    Use LLM to generate hypotheses based on ALL combined results.
    
    This analyzes all DEGs, pathways, and validated edges to generate
    intelligent mechanistic hypotheses using LLM reasoning.
    
    Args:
        perturbation: Perturbation name (can be "X vs Y" for comparisons)
        deg_list: Combined DEGs from all perturbations
        pathways: Top pathways (p <= 0.05)
        validated_edges: Combined validated edges
        cell_type: Cell type context
        species: Species context
        context: Full context dict
    
    Returns:
        List of hypothesis dictionaries (statement, mechanism, etc.)
    """
    from .futurehouse_client import get_gemini_model
    import sys
    
    print(f"  [LLM] Initializing Gemini model...")
    sys.stdout.flush()
    
    try:
        model = get_gemini_model()
        if not model:
            print(f"  [LLM] ⚠️  Gemini model not available, will fall back to template-based hypotheses")
            sys.stdout.flush()
            return []  # Fall back to template-based if LLM unavailable
        print(f"  [LLM] ✓ Model initialized successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"  [LLM] ⚠️  Failed to initialize model: {e}, will fall back to template-based hypotheses")
        sys.stdout.flush()
        return []  # Fall back to template-based if LLM unavailable
    
    # Prepare data summary for LLM
    # Top DEGs (by absolute log2FC)
    top_degs = sorted(deg_list, key=lambda x: abs(x.get("log2fc", 0.0)), reverse=True)[:20]
    significant_degs = [d for d in deg_list if d.get("pval", 1.0) <= 0.05]
    
    # Top pathways
    top_pathways_summary = pathways[:10] if len(pathways) > 10 else pathways
    
    # Summary of validated edges
    strong_edges = [e for e in validated_edges if e.get("confidence", 0.0) >= 0.7][:10]
    
    # Build context for LLM
    data_summary = []
    data_summary.append(f"Perturbation: {perturbation}")
    if cell_type:
        data_summary.append(f"Cell type: {cell_type}")
    if species:
        data_summary.append(f"Species: {species}")
    data_summary.append("")
    
    data_summary.append(f"Differential Expression Analysis:")
    data_summary.append(f"  - Total DEGs analyzed: {len(deg_list)}")
    data_summary.append(f"  - Significantly different genes (p ≤ 0.05): {len(significant_degs)}")
    if top_degs:
        data_summary.append(f"  - Top upregulated genes:")
        for deg in top_degs[:5]:
            if deg.get("log2fc", 0) > 0:
                data_summary.append(f"    • {deg.get('gene', 'Unknown')}: log2FC={deg.get('log2fc', 0.0):.3f}, p={deg.get('pval', 1.0):.3e}")
        data_summary.append(f"  - Top downregulated genes:")
        for deg in sorted(top_degs, key=lambda x: x.get("log2fc", 0.0))[:5]:
            if deg.get("log2fc", 0) < 0:
                data_summary.append(f"    • {deg.get('gene', 'Unknown')}: log2FC={deg.get('log2fc', 0.0):.3f}, p={deg.get('pval', 1.0):.3e}")
    
    data_summary.append("")
    data_summary.append(f"Pathway Enrichment Analysis:")
    data_summary.append(f"  - Total enriched pathways (p ≤ 0.05): {len(pathways)}")
    if top_pathways_summary:
        data_summary.append(f"  - Top enriched pathways:")
        for pathway in top_pathways_summary[:10]:
            name = pathway.get("name", "Unknown")
            nes = pathway.get("NES", 0.0)
            pval = pathway.get("pval") or pathway.get("pvalue") or pathway.get("P-value") or 1.0
            data_summary.append(f"    • {name}: NES={nes:.2f}, p={pval:.3e}")
            member_genes = pathway.get("member_genes", [])
            if member_genes:
                data_summary.append(f"      Genes: {', '.join(member_genes[:5])}")
    
    if strong_edges:
        data_summary.append("")
        data_summary.append(f"Validated Regulatory Edges:")
        data_summary.append(f"  - Strong edges (confidence ≥ 0.7): {len(strong_edges)}")
        for edge in strong_edges[:5]:
            source = edge.get("source", "")
            target = edge.get("target", "")
            direction = edge.get("direction", "unknown")
            confidence = edge.get("confidence", 0.0)
            data_summary.append(f"    • {source} → {target} ({direction}, confidence={confidence:.2f})")
    
    data_context = "\n".join(data_summary)
    
    # Build LLM prompt
    print(f"  [LLM] Preparing data summary for LLM analysis...")
    sys.stdout.flush()
    
    # Determine if this is a comparison question
    is_comparison = perturbation_type == "comparison" and perturbation1 and perturbation2
    
    # Build comparison-specific instructions
    comparison_instructions = ""
    if is_comparison and user_question:
        comparison_instructions = f"""
IMPORTANT: This is a COMPARISON analysis. The user's question is: "{user_question}"

You are comparing TWO SEPARATE perturbations:
- Perturbation 1: {perturbation1}
- Perturbation 2: {perturbation2}

CRITICAL: Generate hypotheses that COMPARE these two perturbations, NOT hypotheses that combine them.
- Do NOT say "{perturbation1} with {perturbation2}" or "{perturbation1} and {perturbation2} together"
- Instead, generate hypotheses that explain which perturbation has stronger effects or different mechanisms
- Focus on comparing their individual effects, not what happens when both are applied simultaneously
- Each hypothesis should address which perturbation has more impact or different biological consequences

Examples of CORRECT hypothesis statements:
- "{perturbation1} perturbation leads to stronger downregulation of X pathway compared to {perturbation2}"
- "{perturbation2} has more pronounced effects on Y signaling than {perturbation1}"
- "{perturbation1} shows greater impact on Z pathway activation"

Examples of INCORRECT hypothesis statements (DO NOT USE):
- "{perturbation1} with {perturbation2} leads to..." (combining them)
- "{perturbation1} perturbation with {perturbation2} at..." (combining them)
- "Combined {perturbation1} and {perturbation2} treatment..." (combining them)

"""
    
    prompt = f"""You are a computational biologist analyzing perturbation data. Based on the comprehensive analysis results below, generate 3-5 testable mechanistic hypotheses.
{comparison_instructions}
Analysis Results:
{data_context}

Please generate mechanistic hypotheses that:
1. Integrate information from differential expression, pathway enrichment, and regulatory edges
2. Propose testable biological mechanisms
3. Are specific and actionable (not vague)
4. Connect the perturbation(s) to downstream effects through pathways and genes
{"5. COMPARE the two perturbations (which has stronger/more effects), do NOT combine them" if is_comparison else ""}

For each hypothesis, provide:
- A clear statement (1-2 sentences){" that compares the two perturbations" if is_comparison else ""}
- A step-by-step mechanism (3-5 steps) explaining how the perturbation leads to the predicted effect
- Key genes/pathways involved

Format your response as a JSON-like structure with hypotheses as a list. Each hypothesis should have:
- "statement": "Clear hypothesis statement{" comparing the two perturbations" if is_comparison else ""}"
- "mechanism": ["Step 1", "Step 2", "Step 3", ...]
- "key_pathways": ["Pathway 1", "Pathway 2", ...]
- "key_genes": ["Gene1", "Gene2", ...]

Response:"""
    
    print(f"  [LLM] Sending request to Gemini API (this may take 30-60 seconds)...")
    sys.stdout.flush()
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"  [LLM] ✓ Received response from Gemini API")
        print(f"  [LLM] Parsing response and extracting hypotheses...")
        sys.stdout.flush()
        
        # Parse LLM response (may be JSON or text)
        # Try to extract JSON if present
        import json
        import re
        
        # Try to find and parse JSON in the response
        # Method 1: Try parsing the entire response as JSON first
        try:
            parsed = json.loads(response_text)
            hypotheses_list = parsed.get("hypotheses", [])
            if hypotheses_list:
                llm_hypotheses = []
                for i, hyp in enumerate(hypotheses_list[:5], 1):
                    if isinstance(hyp, dict):
                        llm_hypotheses.append({
                            "statement": hyp.get("statement", ""),
                            "mechanism": hyp.get("mechanism", []),
                            "key_pathways": hyp.get("key_pathways", []),
                            "key_genes": hyp.get("key_genes", [])
                        })
                if llm_hypotheses:
                    print(f"  [LLM] ✓ Successfully parsed {len(llm_hypotheses)} hypotheses from JSON response")
                    sys.stdout.flush()
                    return llm_hypotheses
        except json.JSONDecodeError:
            pass
        
        # Method 2: Try to extract JSON object from markdown code blocks
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_block_match:
            try:
                parsed = json.loads(json_block_match.group(1))
                hypotheses_list = parsed.get("hypotheses", [])
                if hypotheses_list:
                    llm_hypotheses = []
                    for i, hyp in enumerate(hypotheses_list[:5], 1):
                        if isinstance(hyp, dict):
                            llm_hypotheses.append({
                                "statement": hyp.get("statement", ""),
                                "mechanism": hyp.get("mechanism", []),
                                "key_pathways": hyp.get("key_pathways", []),
                                "key_genes": hyp.get("key_genes", [])
                            })
                    if llm_hypotheses:
                        print(f"  [LLM] ✓ Successfully parsed {len(llm_hypotheses)} hypotheses from JSON code block")
                        sys.stdout.flush()
                        return llm_hypotheses
            except json.JSONDecodeError:
                pass
        
        # Method 3: Try to find the first complete JSON object (handle nested braces properly)
        # Find the first opening brace and try to match it with the corresponding closing brace
        brace_count = 0
        start_idx = response_text.find('{')
        if start_idx != -1:
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        json_str = response_text[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            hypotheses_list = parsed.get("hypotheses", [])
                            if hypotheses_list:
                                llm_hypotheses = []
                                for j, hyp in enumerate(hypotheses_list[:5], 1):
                                    if isinstance(hyp, dict):
                                        llm_hypotheses.append({
                                            "statement": hyp.get("statement", ""),
                                            "mechanism": hyp.get("mechanism", []),
                                            "key_pathways": hyp.get("key_pathways", []),
                                            "key_genes": hyp.get("key_genes", [])
                                        })
                                if llm_hypotheses:
                                    print(f"  [LLM] ✓ Successfully parsed {len(llm_hypotheses)} hypotheses from extracted JSON")
                                    sys.stdout.flush()
                                    return llm_hypotheses
                        except json.JSONDecodeError:
                            pass
                        break
        
        # Fallback: try to parse text response
        # Look for hypothesis statements in the text
        if "hypothesis" in response_text.lower() or "mechanism" in response_text.lower():
            # Try to extract hypotheses from text
            # This is a simple fallback - could be improved
            hypotheses_list = []
            lines = response_text.split("\n")
            current_hyp = None
            
            for line in lines:
                if "statement" in line.lower() or "hypothesis" in line.lower():
                    # Extract statement
                    statement = re.sub(r'^(statement|hypothesis)[:\s]*', '', line, flags=re.IGNORECASE).strip()
                    if statement and len(statement) > 10:
                        current_hyp = {"statement": statement, "mechanism": [], "key_pathways": [], "key_genes": []}
                        hypotheses_list.append(current_hyp)
                elif "mechanism" in line.lower() and current_hyp:
                    continue  # Skip mechanism header
                elif current_hyp and (line.strip().startswith("-") or line.strip().startswith("•") or re.match(r'^\d+\.', line.strip())):
                    # This is a mechanism step
                    step = re.sub(r'^[-•\d\.\s]+', '', line).strip()
                    if step:
                        current_hyp["mechanism"].append(step)
            
            if hypotheses_list:
                print(f"  [LLM] ✓ Successfully parsed {len(hypotheses_list)} hypotheses from text response")
                sys.stdout.flush()
                return hypotheses_list
        
        print(f"  [LLM] ⚠️  Could not parse hypotheses from LLM response")
        sys.stdout.flush()
    except Exception as e:
        # If LLM fails, return empty list (will fall back to template-based)
        print(f"  [LLM] ✗ LLM hypothesis generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return []
    
    print(f"  [LLM] ⚠️  No hypotheses extracted from LLM response")
    sys.stdout.flush()
    return []


def _enrich_hypothesis_with_literature(
    hypothesis_dict: Dict[str, Any],
    hypothesis_id: str,
    perturbation: str,
    cell_type: str,
    validated_edges: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Enrich an LLM-generated hypothesis with literature search and other components.
    
    Args:
        hypothesis_dict: LLM-generated hypothesis dict with statement, mechanism, etc.
        hypothesis_id: Hypothesis ID (e.g., "H1")
        perturbation: Perturbation name
        cell_type: Cell type
        validated_edges: Validated edges
    
    Returns:
        Fully enriched hypothesis dict matching the expected format
    """
    statement = hypothesis_dict.get("statement", "")
    mechanism = hypothesis_dict.get("mechanism", [])
    key_pathways = hypothesis_dict.get("key_pathways", [])
    key_genes = hypothesis_dict.get("key_genes", [])
    
    if not statement:
        return None
    
    # Build literature search query based on hypothesis content
    query_parts = [perturbation]
    # Use hypothesis statement keywords if available
    if statement:
        # Extract key terms from statement (simple approach)
        words = statement.split()
        key_terms = [w.strip('.,;:()[]') for w in words if len(w) > 4 and w.isalpha()][:3]
        query_parts.extend(key_terms)
    if key_pathways:
        query_parts.extend(key_pathways[:2])
    if key_genes:
        query_parts.extend(key_genes[:3])
    if cell_type:
        query_parts.append(cell_type)
    
    query = " ".join([q for q in query_parts if q])
    
    import sys
    print(f"      [Literature] Building search query: {query[:80]}...")
    sys.stdout.flush()
    
    # Perform literature search via Edison Scientific (PaperQA)
    print(f"      [Literature] Searching Edison Scientific (PaperQA) API...")
    sys.stdout.flush()
    literature_papers = search_literature(query, k=10) if query else []
    
    print(f"      [Literature] Found {len(literature_papers)} papers")
    sys.stdout.flush()
    
    # Classify literature support
    print(f"      [Literature] Classifying literature support...")
    sys.stdout.flush()
    pathway_name = key_pathways[0] if key_pathways else ""
    literature_support = _classify_literature_support(
        papers=literature_papers,
        pathway=pathway_name,
        perturbation=perturbation
    )
    
    print(f"      [Literature] Support level: {literature_support.get('overall', 'unknown')}")
    sys.stdout.flush()
    
    # Generate predicted readouts
    predicted_readouts = []
    if key_pathways:
        for pathway_name in key_pathways[:3]:
            if "apoptosis" in pathway_name.lower():
                predicted_readouts.append("Caspase-3/7 activity")
                predicted_readouts.append("Annexin V staining")
            elif "cell cycle" in pathway_name.lower():
                predicted_readouts.append("Cell cycle analysis (PI staining)")
            elif "proliferation" in pathway_name.lower():
                predicted_readouts.append("Cell count")
                predicted_readouts.append("Ki-67 staining")
    
    if not predicted_readouts:
        predicted_readouts = [
            "RNA-seq transcriptome profiling",
            "Protein expression (Western blot or mass spectrometry)",
            "Cell viability assay"
        ]
    
    # Suggest experiments
    experiments = []
    base_exp = f"Perform {perturbation} in {cell_type} cells"
    if key_pathways:
        experiments.append(f"{base_exp} and measure pathway activity for {key_pathways[0]} at 24h")
    if key_genes:
        experiments.append(f"Validate {key_genes[0]} expression using qRT-PCR or RNA-seq")
    if not experiments:
        experiments.append(f"{base_exp} and perform transcriptome profiling (RNA-seq) at 24h")
    
    # Speculation notes
    speculation_notes = []
    if validated_edges:
        speculation_notes.append(f"Directly supported by {len([e for e in validated_edges if e.get('confidence', 0) >= 0.7])} validated regulatory edges.")
    if literature_papers:
        if len(literature_papers) >= 3:
            speculation_notes.append(f"Literature support: {len(literature_papers)} relevant papers found.")
        else:
            speculation_notes.append("Limited direct literature evidence - hypothesis requires experimental validation.")
    else:
        speculation_notes.append("This hypothesis is largely inferred from computational predictions and requires strong experimental validation.")
    
    return {
        "id": hypothesis_id,
        "statement": statement,
        "mechanism": mechanism if isinstance(mechanism, list) else [str(m) for m in mechanism],
        "phenotype_support": None,  # No phenotypes (PhenotypeKB removed)
        "predicted_readouts": predicted_readouts[:5],
        "literature_support": literature_support,
        "experiments": experiments[:4],
        "speculation_notes": " ".join(speculation_notes) if speculation_notes else "Hypothesis combines computational predictions with limited literature evidence."
    }


def _generate_no_hypotheses_explanation(
    perturbation: str,
    top_phenotypes: List[Dict[str, Any]],
    top_pathways: List[Dict[str, Any]],
    deg_list: List[Dict[str, Any]],
    pathways: List[Dict[str, Any]],
    phenotypes: List[Dict[str, Any]],
    literature_papers: List[Dict[str, Any]]
) -> str:
    """Generate LLM-powered explanation for why no hypotheses were generated."""
    from .futurehouse_client import get_gemini_model
    
    # Build context for LLM
    context_parts = []
    
    context_parts.append(f"Perturbation: {perturbation}")
    
    if deg_list:
        significant_degs = [d for d in deg_list if d.get("pval", 1.0) <= 0.05]
        context_parts.append(f"Found {len(significant_degs)} significantly differentially expressed genes (p ≤ 0.05) out of {len(deg_list)} total")
        if significant_degs:
            top_genes = sorted(significant_degs, key=lambda x: abs(x.get("log2fc", 0.0)), reverse=True)[:5]
            context_parts.append(f"Top DEGs: {', '.join([d['gene'] for d in top_genes])}")
    else:
        context_parts.append("No differential expression data available")
    
    if pathways:
        # Count pathways with p-value <= 0.05
        sig_pathways = []
        for p in pathways:
            pval = p.get("pval") or p.get("pvalue") or p.get("P-value") or p.get("P_value") or 1.0
            if isinstance(pval, (int, float)) and pval <= 0.05:
                sig_pathways.append(p)
        context_parts.append(f"Found {len(sig_pathways)} pathways with p ≤ 0.05 out of {len(pathways)} total pathways")
        if sig_pathways:
            top_pathway = sorted(sig_pathways, key=lambda x: abs(x.get("NES", 0.0)), reverse=True)[0]
            context_parts.append(f"Top pathway: {top_pathway.get('name', 'Unknown')} (NES={top_pathway.get('NES', 0.0):.2f})")
    else:
        context_parts.append("No pathway enrichment data available")
    
    if top_phenotypes:
        context_parts.append(f"Found {len(top_phenotypes)} phenotypes with score ≥ 0.6")
        context_parts.append(f"Top phenotype: {top_phenotypes[0].get('name', 'Unknown')} (score={top_phenotypes[0].get('score', 0.0):.2f})")
    else:
        context_parts.append("No phenotypes found with score ≥ 0.6 (minimum threshold for hypothesis generation)")
        if phenotypes:
            context_parts.append(f"Found {len(phenotypes)} phenotypes total, but none met the significance threshold")
    
    context_parts.append(f"Literature search found {len(literature_papers)} potentially relevant papers")
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are analyzing why no mechanistic hypotheses were generated for a perturbation analysis.

Context:
{context}

Please provide a clear, concise explanation (2-3 sentences) for why no hypotheses were generated. Focus on:
1. What data was available vs. what was missing
2. Why the available data didn't meet the thresholds for hypothesis generation
3. What would be needed to generate hypotheses (e.g., stronger pathway enrichments, phenotype predictions, etc.)

Explanation:"""
    
    try:
        model = get_gemini_model()
        if model:
            response = model.generate_content(prompt)
            explanation = response.text.strip()
            return explanation
    except Exception as e:
        # Fallback to rule-based explanation
        pass
    
    # Fallback explanation
    reasons = []
    if not top_phenotypes:
        reasons.append("No phenotypes met the significance threshold (score ≥ 0.6)")
    if not top_pathways:
        reasons.append("No pathways met the significance threshold (p ≤ 0.05)")
    if not deg_list:
        reasons.append("No differential expression data available")
    
    if reasons:
        return f"No hypotheses were generated because: {'; '.join(reasons)}. Available data did not meet the minimum thresholds for generating testable mechanistic hypotheses."
    else:
        return "No hypotheses were generated despite having some data. The available evidence was not sufficient to formulate testable mechanistic hypotheses above the significance thresholds."

