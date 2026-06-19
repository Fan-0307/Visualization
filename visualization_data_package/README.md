# Visualization Data Package

This folder is for large runtime data that should not be committed to Git.

After receiving and extracting the data package, copy or keep the included folders at these project-relative paths:

```text
Visualization/
├── public/
│   ├── data/
│   │   ├── attn/
│   │   └── layer_evo/
│   └── img/
└── src/
    └── data/
        └── processed_data/
            └── photos/
                ├── qwen/
                ├── llava/
                ├── salesforce/
                └── openai/
```

Required data:

```text
public/data/attn/                  Token-level attention JSON files
public/data/layer_evo/             Layer-evolution runtime files
public/img/                        COCO/VQA images served by Vite
src/data/processed_data/photos/    Heatmap images for qwen, llava, salesforce, openai
```

The small metadata files below stay in Git and do not need to be copied separately:

```text
src/data/processed_data/results.json
src/data/processed_data/summary.json
```

If the package is sent as split files, put all parts in the same directory and rebuild the archive first:

```bash
cat visualization_data.tar.gz.part-* > visualization_data.tar.gz
tar -xzf visualization_data.tar.gz -C /path/to/Visualization
```

Then start the app:

```bash
npm install
npm run dev -- --host 0.0.0.0
```
