import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("docs/data/ranking_report.json", "r", encoding="utf-8") as f:
    data = json.load(f)

genomics = data.get("categories", {}).get("Genomics", {})

targets = ["espectre", "wasmedge", "dapr"]

for track in ["top_20_pipelines", "top_10_utilities"]:
    items = genomics.get(track, [])
    for p in items:
        fn = p.get("full_name", "").lower()
        if any(t in fn for t in targets):
            print(f"=== FOUND in {track} ===")
            print(f"full_name: {p.get('full_name')}")
            print(f"description: {p.get('description', '')}")
            print(f"topics: {p.get('topics', [])}")

            desc = (p.get("description") or "").lower()
            topics_list = [t.lower() for t in p.get("topics", [])]
            full_text = f"{desc} {' '.join(topics_list)}"

            NON_BIO_BLACKLIST = [
                "wasmedge", "dapr", "espectre", "runtime", "orchestration", "kubernetes", "k8s",
                "service mesh", "microservice", "serverless", "cloud native", "devops",
                "web framework", "frontend", "backend", "react", "vue", "angular",
                "game engine", "blockchain", "cryptocurrency", "machine learning platform",
                "deep learning framework", "chatbot", "llm", "large language model"
            ]
            BIO_SAFELIST = [
                "bioinformatics", "genomics", "transcriptomics", "proteomics", "metabolomics",
                "metagenomics", "epigenetics", "sequencing", "alignment", "variant",
                "gene", "genome", "rna", "dna", "protein", "cell", "single-cell",
                "ngs", "chip-seq", "atac-seq", "methylation", "expression",
                "phylogenetic", "taxonomy", "microbiome", "omics", "biomarker",
                "scrna", "spatial transcriptomics", "chromatin"
            ]

            has_bio = any(term in full_text for term in BIO_SAFELIST)
            has_blacklist = any(term in full_text for term in NON_BIO_BLACKLIST)

            print(f"full_text: {full_text[:200]}")
            print(f"has_bio_safelist: {has_bio}")
            if has_bio:
                matched = [t for t in BIO_SAFELIST if t in full_text]
                print(f"  matched bio terms: {matched}")
            print(f"has_blacklist: {has_blacklist}")
            if has_blacklist:
                matched = [t for t in NON_BIO_BLACKLIST if t in full_text]
                print(f"  matched blacklist terms: {matched}")
            print()
