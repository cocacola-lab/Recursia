# Component inventory for the standalone FR3 subset

This subset is based on NVIDIA ASPIRE. Its project license and copyright notice
are preserved in ../LICENSE and ../NOTICE. Component-specific terms remain applicable.

## Included inherited material

- ASPIRE FR3 runtime and RSI adaptations: upstream notices are preserved.
  recursia/real/cap/agent/rsi/simulation/SOURCE.json records the ported simulation components.
- CaP-X-derived framework material: the upstream MIT text is retained in MIT-CaP-X.txt.
- Hydra-derived utility attribution: the upstream MIT text is retained in MIT-Hydra.txt.
- Contact-GraspNet compatibility patch: adapted from ASPIRE's simulation patch.
  The patch does not relicense the external implementation or its weights.

- Franka FR3 MJCF and its referenced meshes: bundled under
  `recursia/real/robot/franka/model/`, with the original Apache-2.0 license.
  Public model provenance is documented in that directory.

## External installations, not redistributed

- SAM3 source/model: https://github.com/facebookresearch/sam3 and its model terms.
- Contact-GraspNet PyTorch backend: https://github.com/elchun/contact_graspnet_pytorch
  (the installer pins a revision).
- franky-control and its libfranka runtime: installed from public Python distributions.
- Python dependencies: installed through their own distributions.

No learned-model checkpoint, private SDK license file, or dataset is included.
The public Recursia paper is provided in assets/Recursia.pdf; selected physical
demonstration videos are provided in assets/media/. The retained THIRD_PARTY_NOTICES.md describes
additional parent-repository components; those components are not included in this subset.
