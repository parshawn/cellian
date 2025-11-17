"""Script to download KEGG pathway data for the reasoning engine."""
import os
import json
import requests
from typing import Dict, List
from pathlib import Path


def download_kegg_pathway(pathway_id: str) -> Dict:
    """
    Download a single KEGG pathway.
    
    Args:
        pathway_id: KEGG pathway ID (e.g., "hsa04630")
    
    Returns:
        Dictionary with pathway data
    """
    base_url = "https://rest.kegg.jp"
    
    # Get pathway info
    info_url = f"{base_url}/get/{pathway_id}"
    try:
        response = requests.get(info_url, timeout=30)
        response.raise_for_status()
        info_text = response.text
    except Exception as e:
        print(f"Warning: Could not download pathway info for {pathway_id}: {e}")
        return None
    
    # Get pathway genes
    link_url = f"{base_url}/link/hsa/{pathway_id}"
    try:
        response = requests.get(link_url, timeout=30)
        response.raise_for_status()
        link_text = response.text
    except Exception as e:
        print(f"Warning: Could not download pathway genes for {pathway_id}: {e}")
        link_text = ""
    
    # Parse pathway name from info
    pathway_name = pathway_id
    for line in info_text.split('\n'):
        if line.startswith('NAME'):
            pathway_name = line.split('NAME')[1].strip()
            break
    
    # Parse genes from link
    genes = []
    for line in link_text.split('\n'):
        if line.strip():
            parts = line.split('\t')
            if len(parts) >= 2:
                gene_id = parts[1].split(':')[-1] if ':' in parts[1] else parts[1]
                genes.append(gene_id)
    
    # For now, create a simple structure
    # In a full implementation, you'd parse the KGML file for edges
    pathway_data = {
        "pathway_id": pathway_id,
        "name": pathway_name,
        "genes": list(set(genes)),  # Remove duplicates
        "edges": [],  # Would be populated from KGML parsing
        "category": "signaling"  # Default, could be parsed from info
    }
    
    return pathway_data


def download_pathways(pathway_ids: List[str], output_file: str):
    """
    Download multiple KEGG pathways and save to JSON.
    
    Args:
        pathway_ids: List of KEGG pathway IDs
        output_file: Path to output JSON file
    """
    pathways = {}
    
    print(f"Downloading {len(pathway_ids)} pathways...")
    for i, pathway_id in enumerate(pathway_ids, 1):
        print(f"  [{i}/{len(pathway_ids)}] Downloading {pathway_id}...", end=" ")
        pathway_data = download_kegg_pathway(pathway_id)
        if pathway_data:
            pathways[pathway_id] = pathway_data
            print(f"✓ ({len(pathway_data['genes'])} genes)")
        else:
            print("✗")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(pathways, f, indent=2)
    
    print(f"\n✓ Saved {len(pathways)} pathways to {output_file}")
    return pathways


def main():
    """Main function to download pathway data."""
    # Common pathways relevant to perturbations
    pathway_ids = [
        "hsa04630",  # JAK-STAT signaling pathway
        "hsa04064",  # NF-kappa B signaling pathway
        "hsa04620",  # Toll-like receptor signaling pathway
        "hsa04010",  # MAPK signaling pathway
        "hsa04151",  # PI3K-Akt signaling pathway
        "hsa04621",  # NOD-like receptor signaling pathway
        "hsa04650",  # Natural killer cell mediated cytotoxicity
        "hsa04662",  # B cell receptor signaling pathway
        "hsa04664",  # Fc epsilon RI signaling pathway
        "hsa04666",  # Fc gamma R-mediated phagocytosis
    ]
    
    # Output directory
    data_dir = Path(__file__).parent.parent / "data" / "pathways"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "kegg_pathways.json"
    
    print("KEGG Pathway Data Downloader")
    print("=" * 50)
    print(f"Output directory: {data_dir}")
    print(f"Output file: {output_file}")
    print()
    
    pathways = download_pathways(pathway_ids, str(output_file))
    
    # Print summary
    total_genes = sum(len(p["genes"]) for p in pathways.values())
    unique_genes = len(set(gene for p in pathways.values() for gene in p["genes"]))
    
    print("\nSummary:")
    print(f"  Pathways downloaded: {len(pathways)}")
    print(f"  Total gene entries: {total_genes}")
    print(f"  Unique genes: {unique_genes}")
    print(f"\n✓ Pathway data ready for use!")


if __name__ == "__main__":
    main()

