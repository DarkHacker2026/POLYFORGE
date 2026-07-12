import React from 'react';
import './index.css';

function App() {
  return (
    <div className="pf-app">
      <div className="pf-bg">
        <div className="pf-orb pf-orb-red"></div>
        <div className="pf-orb pf-orb-blue"></div>
      </div>

      <nav className="pf-nav">
        <div className="pf-logo">POLY<span className="pf-red">FORGE</span></div>
        <a href="https://github.com/DarkHacker2026/POLYFORGE" target="_blank" rel="noreferrer" className="pf-nav-link">GitHub →</a>
      </nav>

      <main className="pf-landing">
        <div className="pf-landing-inner">
          <p className="pf-hero-tag">AMD Developer Hackathon · Unicorn Track</p>
          <h1 className="pf-landing-title">
            Any CUDA.<br/>
            Any Hardware.<br/>
            <span className="pf-red">Verified.</span>
          </h1>
          <p className="pf-landing-desc">
            POLYFORGE is an AI-powered CUDA-to-hardware compiler pipeline. Drop in a CUDA kernel,
            and our LLM comprehends it, our Zero-Trust Oracle mathematically proves it, and our
            compiler lowers it to any architecture — RISC-V GPUs, ARM, x86, and beyond.
          </p>

          <div className="pf-landing-features">
            <div className="pf-landing-feature">
              <span className="pf-landing-icon">🧠</span>
              <h3>LLM Comprehension</h3>
              <p>Fireworks AI Kimi-2.6 reads raw CUDA and understands thread indexing, synchronization, and memory patterns.</p>
            </div>
            <div className="pf-landing-feature">
              <span className="pf-landing-icon">🛡️</span>
              <h3>Zero-Trust Oracle</h3>
              <p>Independent Clang AST simulation verifies all threads. Catches LLM hallucinations and data races before touching silicon.</p>
            </div>
            <div className="pf-landing-feature">
              <span className="pf-landing-icon">⚙️</span>
              <h3>Hardware Lowering</h3>
              <p>Write CUDA once, run anywhere. Verified kernels lowered to RISC-V GPUs, ARM, x86, and custom silicon.</p>
            </div>
          </div>

          <div className="pf-landing-btns">
            <a href="https://github.com/DarkHacker2026/POLYFORGE" target="_blank" rel="noreferrer" className="pf-btn pf-btn-red">
              View on GitHub →
            </a>
          </div>

          <div className="pf-landing-tech">
            <span>FastAPI</span><span>·</span>
            <span>React</span><span>·</span>
            <span>Vortex RISC-V</span><span>·</span>
            <span>Fireworks AI</span><span>·</span>
            <span>Clang AST</span>
          </div>
        </div>
      </main>

      <footer className="pf-footer">
        <p>Built for the AMD Developer Hackathon ACT II — Unicorn Track</p>
      </footer>
    </div>
  );
}

export default App;