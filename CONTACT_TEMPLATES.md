# Contact Templates

Three ready-to-send outreach templates for the parallel-oracle project.

---

## Template 1 — Vortex Georgia Tech (Blaise Tine)

**To:** blaise.tine@ece.gatech.edu  
**Subject:** Race-detecting oracle for Vortex kernels — free analysis offer

---

Hi Blaise,

We're a team that has been building a race-detecting oracle for parallel kernels targeting custom hardware. Unlike static analysis tools that assume a specific memory model, our oracle is grounded in empirically-discovered hardware facts — we probed Vortex directly using `rtlsim` to learn its actual synchronization semantics, then built a reference model on top of those observations. The oracle catches RAW, WAW, and WAR data races at the IR level before any silicon time is spent, and it's completely decoupled from any surface language or ISA.

We'd love to run your existing kernel benchmarks through it. The process on your end is minimal — your compiler just needs to emit a simple READ/WRITE/BARRIER JSON IR per thread (we have a one-page guide for that). We'll return a full race-analysis report: which kernels pass, which have hazards, and exactly which thread/address/epoch combinations are in conflict. If this would be useful for validating your benchmark suite, we're happy to set up a quick call or just exchange kernel files over email. Would this be useful for verifying your benchmark suite?

Best,  
[Your Name] — Hackathon Project Team  
[your-repo-url]

---

## Template 2 — CHIPYARD / Gemmini Team (Cold Email)

**To:** [Gemmini maintainer email]  
**Subject:** Parallel kernel race oracle — useful for Gemmini workloads?

---

Hi [Name],

We've built an open-source tool called **parallel-oracle** that verifies parallel memory access patterns for custom accelerators and hardware compilers. It detects data races (RAW, WAW, WAR) by simulating thread-level memory accesses against a synchronization model, and it works from a simple READ/WRITE/BARRIER IR that any compiler backend can emit in a few lines of code. It requires no assumptions about a specific ISA, cache model, or runtime — making it a natural fit for accelerators like Gemmini where the memory hierarchy and dataflow are custom.

We're offering free race analysis for any team willing to share a kernel or workload. If you have existing DNN layer kernels or matrix-operation benchmarks in Gemmini's compiler pipeline, we can take their memory access traces, convert them to our IR, and return a diagnostic report showing whether any cross-thread hazards exist in the access pattern. There's no obligation beyond sharing the kernel description. Happy to elaborate on the IR format or share our results from probing other RISC-V targets. Would this be worth a quick exchange?

Best,  
[Your Name] — Hackathon Project Team  
[your-repo-url]

---

## Template 3 — GitHub Issue for Open-Source RISC-V GPU Projects

**Title:** Request: run your parallel kernels through our race-detection oracle

---

**Body:**

Hi maintainers 👋

We've built an open-source tool called **parallel-oracle** — a race-detecting oracle for parallel kernels targeting custom hardware. It checks for RAW, WAW, and WAR data races by simulating thread-level memory accesses and barrier synchronization, without assuming any particular ISA, cache model, or runtime environment.

**How it works in one sentence:** your compiler emits a simple JSON IR (READ, WRITE, BARRIER per thread), and the oracle tells you whether any cross-thread memory hazard exists in that access pattern — with a full diagnostic showing thread IDs, addresses, access types, and sync epochs.

**Why this might be useful for your project:**
- If you have an existing benchmark suite or test kernels, we can run them through the oracle and return a free race-analysis report.
- It works independently of your RTL simulator — useful as a fast pre-RTL regression step in CI.
- The IR is simple enough to emit from any compiler backend in a day of integration work.

📄 **Full documentation:** [README_ORACLE.md](README_ORACLE.md)  
🗺️ **Compiler integration guide:** [OUTREACH_GUIDE.md](OUTREACH_GUIDE.md)

If you have kernel benchmarks you'd be willing to share (even just a single kernel), we'd love to run them and share the results back with you. Feel free to attach JSON IR directly to this issue, or point us to your benchmark directory and we'll handle the IR conversion ourselves using your compiler's memory access trace.

Would anyone on this team be interested?

— [Your Name], Hackathon Project  
[your-repo-url]
