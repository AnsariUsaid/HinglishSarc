import { useState, useEffect, useRef } from 'react';
import Orb from './components/Orb';
import { NotchNav } from './components/ui/NotchNav';
import { ElectricCard } from './components/ui/ElectricCard';
import './App.css';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

/* ───────── Hooks & Utils ───────── */
function useInView(threshold = 0.15) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.unobserve(el); } },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return [ref, visible];
}

function Counter({ end, suffix = '', duration = 2000 }) {
  const [val, setVal] = useState(0);
  const [ref, visible] = useInView(0.3);
  useEffect(() => {
    if (!visible) return;
    const start = performance.now();
    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(eased * end);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [visible, end, duration]);
  return <span ref={ref}>{val.toFixed(end < 10 ? 1 : 0)}{suffix}</span>;
}

/* ───────── Premium Components ───────── */
function GlassStep({ num, title, desc, delay = 0 }) {
  const [ref, visible] = useInView(0.2);
  return (
    <div ref={ref} className={`glass-step ${visible ? 'in' : ''}`} style={{ transitionDelay: `${delay}ms` }}>
      <div className="gs-num">{num}</div>
      <div className="gs-content">
        <h4>{title}</h4>
        <p>{desc}</p>
      </div>
    </div>
  );
}

function DemoTerminal() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [ref, visible] = useInView();

  const examples = [
    'I just love waiting in traffic for two hours.',
    'Mujhe laga tha aaj baarish hogi, thank god nahi hui.',
    'Oh wow, another meeting that could have been an email.',
    'Kya mast weather hai, bilkul barbaad kar diya mood.',
  ];

  async function analyze() {
    const payloadText = text.trim();
    if (!payloadText) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: payloadText }),
      });

      if (!response.ok) {
        let detail = `Backend returned ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload?.detail) detail = errorPayload.detail;
        } catch {}
        throw new Error(detail);
      }

      const data = await response.json();
      setResult({
        sarcastic: Boolean(data.sarcastic),
        confidence: Number(data.confidence) || 0,
        emotion: data.emotion || 'neutral',
        trajectory: Array.isArray(data.trajectory) ? data.trajectory : ['neutral', 'neutral', 'neutral'],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to the prediction backend.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div ref={ref} className={`demo-terminal ${visible ? 'in' : ''}`}>
      <div className="term-header">
        <div className="term-dots">
          <span className="dot red"></span>
          <span className="dot yellow"></span>
          <span className="dot green"></span>
        </div>
        <span className="term-title">hinglish-sarcasm-detector</span>
      </div>
      
      <div className="term-body">
        <div className="term-examples">
          <span className="term-comment"># Select a prepared example or type your own:</span>
          {examples.map((ex, i) => (
            <button key={i} className="term-example-btn" onClick={() => { setText(ex); setResult(null); setError(''); }}>
              {ex.slice(0, 35)}…
            </button>
          ))}
        </div>

        <div className="term-input-group">
          <span className="term-prompt">~/predict $</span>
          <input
            className="term-input"
            value={text}
            spellCheck={false}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && analyze()}
            placeholder="Enter Hinglish sentence..."
          />
        </div>
        
        <button className="term-run-btn" onClick={analyze} disabled={loading || !text.trim()}>
          {loading ? '[ RUNNING_INFERENCE... ]' : '[ EXECUTE_ANALYSIS ]'}
        </button>

        {error && <div className="term-output error">[ERROR]: {error}</div>}

        {result && (
          <div className="term-output success">
            <div className="output-line">
              <span className="key">classification:</span> 
              <span className={`val ${result.sarcastic ? 'sarc' : 'norm'}`}>
                {result.sarcastic ? '"Sarcastic"' : '"Not Sarcastic"'}
              </span>
            </div>
            <div className="output-line">
              <span className="key">confidence:</span> <span className="val num">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="output-line">
              <span className="key">emotion:</span> <span className="val str">"{result.emotion}"</span>
            </div>
            <div className="output-line">
              <span className="key">trajectory:</span> 
              <span className="val arr">
                [{result.trajectory.map((t, i) => <span key={i}>"{t}"{i < result.trajectory.length-1 ? ', ' : ''}</span>)}]
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═════════════════════════════════════════════ */
/*                   APP                        */
/* ═════════════════════════════════════════════ */
export default function App() {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <div className="app-container">
      {/* ──── NAV ──── */}
      <nav className={`navbar ${scrollY > 60 ? 'scrolled' : ''}`}>
        <div className="nav-inner">
          <a href="#hero" className="nav-logo">HinglishSarc</a>
          <NotchNav
            items={[
              { value: "hero", label: "Overview", href: "#hero" },
              { value: "features", label: "Capabilities", href: "#features" },
              { value: "pipeline", label: "Architecture", href: "#pipeline" },
              { value: "results", label: "Evaluation", href: "#results" },
              { value: "demo", label: "Console", href: "#demo" }
            ]}
            defaultValue="hero"
            ariaLabel="Main navigation"
          />
        </div>
      </nav>

      {/* ──── HERO (ORB) ──── */}
      <section id="hero" className="hero">
        <div className="orb-wrapper">
          <Orb
            hoverIntensity={2}
            rotateOnHover={true}
            hue={0}
            forceHoverState={false}
            backgroundColor="#000000"
          />
        </div>
        
        <div className="hero-content" style={{ opacity: Math.max(0, 1 - scrollY * 0.002) }}>
          <div className="badge">IEEE Research</div>
          <h1>Emotion-Aware<br/>Sarcasm Detection</h1>
          <p>Leveraging multilingual BERT and emotion trajectory analysis to detect sarcasm in code-mixed social media interactions.</p>
          <div className="hero-actions">
            <a href="#demo" className="btn btn-primary">Try Live Demo</a>
            <a href="#pipeline" className="btn btn-outline">Read Methodology</a>
          </div>
        </div>
      </section>

      <div className="content-wrapper">
        {/* ──── FEATURES ──── */}
        <section id="features" className="section features-section">
          <div className="section-header">
            <h2>Capabilities</h2>
            <p>Designed for the complex linguistic nature of Hindi-English code-switching.</p>
          </div>
          
          <div className="bento-grid">
            <div className="bento-cell hero-cell">
              <div className="bento-content">
                <span className="b-stat"><Counter end={91.6} suffix="%" /></span>
                <h4>Macro-F1 Score</h4>
                <p>Rigorous stratified evaluation across extremely imbalanced datasets.</p>
              </div>
            </div>
            <div className="bento-cell small-cell">
              <div className="bento-content">
                <span className="b-icon">🎭</span>
                <h4>Emotion Trajectory</h4>
                <p>Maps affective incongruity—the core hallmark of sarcasm.</p>
              </div>
            </div>
            <div className="bento-cell small-cell">
              <div className="bento-content">
                <span className="b-icon">🌐</span>
                <h4>Multilingual mBERT</h4>
                <p>Fine-tuned to naturally handle transliteration patterns and script toggling.</p>
              </div>
            </div>
            <div className="bento-cell wide-cell">
              <div className="bento-content row-layout">
                <div className="b-text">
                  <h4>Real-Time API</h4>
                  <p>Optimized FastAPI backend delivering asynchronous WebGL inference endpoints.</p>
                </div>
                <div className="b-visual abstract-glow"></div>
              </div>
            </div>
          </div>
        </section>

        {/* ──── PIPELINE ──── */}
        <section id="pipeline" className="section pipeline-section">
          <div className="section-header">
            <h2>Pipeline Architecture</h2>
            <p>End-to-end processing from raw tweet to classified inference.</p>
          </div>

          <div className="glass-flow-container">
            <div className="glass-flow-line"></div>
            <div className="glass-flow-grid">
              <GlassStep num="1" title="Data Preprocessing" desc="Hinglish social media text is cleaned. Hashtags, URLs, and noise removed. Stratified 70/15/15 sampling applied." delay={0} />
              <GlassStep num="2" title="Emotion Labeling" desc="Text annotated with emotion states (joy, anger, sadness, fear, surprise, neutral) to capture affective shifts." delay={100} />
              <GlassStep num="3" title="Transformer Fine-Tuning" desc="bert-base-multilingual-cased trained for 5 epochs using AdamW, linear warmup, and class-weighted loss." delay={200} />
              <GlassStep num="4" title="Inference Output" desc="Live model executes binary classification, delivering confidence scoring and granular emotion trajectories." delay={300} />
            </div>
          </div>
        </section>

        {/* ──── RESULTS ──── */}
        <section id="results" className="section results-section">
          <div className="section-header">
            <h2>Evaluation Outcomes</h2>
            <p>Benchmarked across standard and imbalanced domains.</p>
          </div>

          <div className="sleek-results">
            <ElectricCard className="sleek-result-block" color="#00aaff">
              <div className="sr-value"><Counter end={91.6} suffix="%" /></div>
              <div className="sr-title">Hinglish Dataset F1</div>
              <div className="sr-divider"></div>
              <div className="sr-stats">
                <span>Acc: 91.6%</span>
                <span>n=8,946</span>
              </div>
            </ElectricCard>

            <ElectricCard className="sleek-result-block dim" color="#00aaff">
              <div className="sr-value"><Counter end={73.4} suffix="%" /></div>
              <div className="sr-title">Hindi Subset F1</div>
              <div className="sr-divider"></div>
              <div className="sr-stats">
                <span>Acc: 86.1%</span>
                <span>n=1,106</span>
              </div>
            </ElectricCard>
          </div>
        </section>

        {/* ──── DEMO ──── */}
        <section id="demo" className="section demo-section">
          <div className="section-header">
            <h2>Console Interface</h2>
            <p>Interact directly with the deployed inference model.</p>
          </div>
          
          <DemoTerminal />
        </section>
      </div>

      {/* ──── FOOTER ──── */}
      <footer className="footer">
        <div className="footer-content">
          <div className="brand">HinglishSarc</div>
          <div className="links">
            <a href="https://github.com/AnsariUsaid/HinglishSarc" target="_blank" rel="noreferrer">GitHub</a>
            <a href="#pipeline">Architecture</a>
          </div>
        </div>
        <div className="footer-bottom">© 2025 HinglishSarc. IEEE Conference Paper Prototype.</div>
      </footer>
    </div>
  );
}
