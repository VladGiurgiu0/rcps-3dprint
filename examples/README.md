# Examples

This folder contains example configurations and small inputs for trying the pipeline.

## Files

- `config_50mm_d6_phi035.yaml` — reference single-tile configuration matching `../data_example/packing.xyzd` (50 × 50 × 50 mm tile, 6 mm spheres, target porosity 0.35, 718 spheres).

## Running

After installing the package:

```bash
pip install -e ..
rcps-build config_50mm_d6_phi035.yaml
```

Outputs land in `./out/` next to the config.

## Building a multi-tile facility

The reference geometry from `data_example/matlab_how it was made 2.png` is 200 × 200 × 50 mm = 4 × 4 × 1 tiles. Once `rcps-facility` lands:

```bash
rcps-facility config_50mm_d6_phi035.yaml --grid 4 4 1 --out-dir ./facility_200x200x50/
```

Per-tile `keep_sides` are derived from each tile's position in the grid so the printed tiles dovetail correctly.
