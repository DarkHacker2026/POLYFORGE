# Source Target Status

The original plan named e-GPU as the preferred target. The e-GPU paper says the full repository is available at:

```text
https://github.com/esl-epfl/e-gpu
```

As of this workspace setup, GitHub returns `Repository not found` for that URL and common spelling variants.

The e-GPU paper also states that the e-GPU compute unit is based on Vortex. We therefore use the public Vortex repository as the available open RTL target:

```text
https://github.com/vortexgpgpu/vortex
local: vendor/vortex
commit: b70e0d5
```

This keeps the hackathon prototype honest:

- Current implementation target: Vortex.
- Claimed method: agent-grown compiler backend from open RTL/ISA facts.
- Future target: e-GPU if/when the EPFL repository becomes public.

