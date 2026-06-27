# Attention Data Processors

Recovered processors for generating VLM attention visualization data.

## Layer Evolution

```bash
cd /home/jovyan/Visualization/data_processor/layer_evolution
./run_all.sh --limit 1 --overwrite
./run_all.sh --overwrite
```

Outputs default to:

`/home/jovyan/Visualization/src/data/layer_evolution`

Default display sampling keeps CLIP at 12 real layers and samples other models
to 24 real layers. Use `--save-all-layers` to write every collected real layer.

## Last Layer

After layer evolution data exists:

```bash
cd /home/jovyan/Visualization/data_processor/last_layer
./extract_last_layer.sh
```

This copies each sample's last displayed layer from layer-evolution metadata into
a compact last-layer dataset.
