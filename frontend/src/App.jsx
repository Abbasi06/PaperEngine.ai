import { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Send, Plus, Brain, X, Check, FileText, SlidersHorizontal, PlayCircle, BookOpen, Copy, Headphones, MonitorPlay, Zap, GraduationCap, Atom, Network, FlaskConical, Sigma, Binary, Dna, Microscope, Calculator, Globe, Cpu, Database, Code, User, Beaker, TestTube, Compass, Component, Radio, Variable, Layers } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_URL = "http://localhost:8080";

// --- CUSTOM ICON ---
const SwishPlane = ({ size = 48, color = "#DA7756" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 12L22 2L12 22L10 14L2 12Z" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M22 2L10 14" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M6 18C4 16 3 14 4 10C5 6 9 5 12 7" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="2 3" opacity="0.6" />
  </svg>
);

const FloatingBackground = () => {
  const icons = [
    { Icon: Atom, color: "#DA7756", size: 48, top: "10%", left: "22%", delay: "0s" },
    { Icon: Globe, color: "#06b6d4", size: 44, top: "18%", right: "5%", delay: "1.2s" },
    { Icon: Network, color: "#6366f1", size: 40, top: "28%", left: "4%", delay: "2s" },
    { Icon: Database, color: "#f97316", size: 38, top: "35%", right: "24%", delay: "4.5s" },
    { Icon: Beaker, color: "#ef4444", size: 34, top: "48%", left: "20%", delay: "1.2s" },
    { Icon: Brain, color: "#DA7756", size: 45, top: "55%", right: "6%", delay: "4.2s" },
    { Icon: Microscope, color: "#ef4444", size: 42, top: "68%", left: "5%", delay: "0.5s" },
    { Icon: Dna, color: "#ec4899", size: 50, top: "75%", right: "20%", delay: "4s" },
    { Icon: FileText, color: "#64748b", size: 38, top: "85%", left: "15%", delay: "2.5s" },
    { Icon: Calculator, color: "#3b82f6", size: 30, top: "92%", right: "8%", delay: "3.5s" },
  ];

  return (
    <div className="floating-bg" style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
      <style>{`
        @keyframes float { 
          0% { transform: translateY(0px) rotate(0deg); } 
          33% { transform: translateY(-30px) rotate(5deg); } 
          66% { transform: translateY(10px) rotate(-5deg); } 
          100% { transform: translateY(0px) rotate(0deg); } 
        }
        .float-icon { animation: float 12s ease-in-out infinite; opacity: 0.12; position: absolute; }
      `}</style>
      {icons.map((item, i) => (
        <item.Icon key={i} className="float-icon" size={item.size} style={{
          color: item.color, top: item.top, left: item.left, right: item.right, animationDelay: item.delay
        }} />
      ))}
    </div>
  );
};

const QuizGame = ({ data }) => {
  const [selected, setSelected] = useState({});
  const questions = data.quiz || [];
  const handleSelect = (qIdx, opt) => { if (!selected[qIdx]) setSelected({ ...selected, [qIdx]: opt }); };

  return (
    <div className="quiz-container">
      {questions.map((q, i) => {
        const userAnswer = selected[i];
        const isCorrect = userAnswer === q.answer;
        return (
          <div key={i} className="quiz-card">
            <h4>{i + 1}. {q.question}</h4>
            <div className="options-list">
              {q.options.map((opt, idx) => (
                <button key={idx} 
                  className={`quiz-option ${userAnswer ? (opt===q.answer?'correct':(opt===userAnswer?'wrong':'disabled')) : ''}`}
                  onClick={() => handleSelect(i, opt)}>{opt}
                </button>
              ))}
            </div>
            {userAnswer && <div className={`feedback ${isCorrect?'text-green':'text-red'}`}>{isCorrect?"✅ Correct!":`❌ Answer: ${q.answer}`}</div>}
          </div>
        );
      })}
    </div>
  );
};

const ArtifactViewer = ({ artifact, onClose }) => {
  if (!artifact) return null;
  return (
    <div className="modal-overlay">
      <div className="modal-card wide">
        <div className="modal-header">
          <div className="header-title">{artifact.type.includes("Quiz")?<PlayCircle size={20}/>:<BookOpen size={20}/>}<h2>{artifact.type}</h2></div>
          <button onClick={onClose}><X size={24}/></button>
        </div>
        <div className="modal-body scrollable">{artifact.type.includes("Quiz") ? <QuizGame data={artifact.data}/> : <div className="markdown-body"><ReactMarkdown>{typeof artifact.data==='string'?artifact.data:JSON.stringify(artifact.data,null,2)}</ReactMarkdown></div>}</div>
      </div>
    </div>
  );
};

const SurveyModal = ({ isOpen, onClose, onSave }) => {
  const [style, setStyle] = useState("Reading");
  const [depth, setDepth] = useState("Layman");
  if (!isOpen) return null;

  const learningStyles = [
    { id: 'Reading', icon: BookOpen, label: 'Reading Books', desc: 'Detailed text summaries & articles.' },
    { id: 'Watching', icon: MonitorPlay, label: 'YouTube Tutorials', desc: 'Visual explanations & video scripts.' },
    { id: 'Listening', icon: Headphones, label: 'Podcasts', desc: 'Audio scripts & conversational briefs.' },
    { id: 'All', icon: Layers, label: 'Full Spectrum', desc: 'Generates every learning material available.' },
  ];

  const depthLevels = [
    { id: 'Layman', icon: Zap, label: 'The Big Picture', desc: 'Simple analogies, clear language, zero jargon.' },
    { id: 'Researcher', icon: GraduationCap, label: 'Academic Deep Dive', desc: 'Technical rigor, citations, and raw data analysis.' },
  ];

  return (
    <div className="modal-overlay">
      <div className="modal-card" style={{ maxWidth: '800px', width: '90%', padding: '2rem' }}>
        <div className="modal-header" style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem' }}>Customize Your Engine</h2>
          <button onClick={onClose} className="close-btn"><X size={24}/></button>
        </div>
        <div className="modal-body">
          <div className="section" style={{ marginBottom: '2rem' }}>
            <label style={{ display: 'block', marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '600', color: '#334155' }}>1. What is your preferred learning style?</label>
            <div className="options-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
              {learningStyles.map((item) => (
                <button key={item.id} 
                  className={`option-btn ${style===item.id?'selected':''}`} 
                  onClick={()=>setStyle(item.id)}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center',
                    padding: '1.5rem', borderRadius: '1rem', border: style===item.id ? '2px solid #DA7756' : '1px solid #e2e8f0',
                    backgroundColor: style===item.id ? '#FFF0EB' : 'white', transition: 'all 0.2s', cursor: 'pointer'
                  }}
                >
                  <item.icon size={32} style={{ marginBottom: '1rem', color: style===item.id ? '#DA7756' : '#94a3b8' }}/>
                  <div style={{ fontWeight: '700', fontSize: '1rem', marginBottom: '0.5rem', color: style===item.id ? '#DA7756' : '#1e293b' }}>{item.label}</div>
                  {item.features ? (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', justifyContent: 'center' }}>
                      {item.features.map(f => (
                        <span key={f} style={{ fontSize: '0.7rem', padding: '0.25rem 0.5rem', borderRadius: '0.35rem', backgroundColor: style===item.id ? 'rgba(218,119,86,0.15)' : '#f1f5f9', color: style===item.id ? '#DA7756' : '#64748b', fontWeight: '600' }}>{f}</span>
                      ))}
                    </div>
                  ) : <div style={{ fontSize: '0.85rem', color: '#64748b', lineHeight: '1.4' }}>{item.desc}</div>}
                </button>
              ))}
            </div>
          </div>
          <div className="section">
            <label style={{ display: 'block', marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '600', color: '#334155' }}>2. How deep should we go?</label>
            <div className="options-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
              {depthLevels.map((item) => (
                <button key={item.id} 
                  className={`option-btn ${depth===item.id?'selected':''}`} 
                  onClick={()=>setDepth(item.id)}
                  style={{
                    display: 'flex', flexDirection: 'row', alignItems: 'flex-start', textAlign: 'left',
                    padding: '1.5rem', borderRadius: '1rem', border: depth===item.id ? '2px solid #DA7756' : '1px solid #e2e8f0',
                    backgroundColor: depth===item.id ? '#FFF0EB' : 'white', transition: 'all 0.2s', cursor: 'pointer', gap: '1rem'
                  }}
                >
                  <div style={{ padding: '0.5rem', borderRadius: '0.5rem', backgroundColor: depth===item.id ? '#FAD4C8' : '#f1f5f9', color: depth===item.id ? '#DA7756' : '#94a3b8' }}>
                    <item.icon size={24}/>
                  </div>
                  <div>
                    <span style={{ display: 'block', fontWeight: '700', fontSize: '1rem', marginBottom: '0.25rem', color: depth===item.id ? '#DA7756' : '#1e293b' }}>{item.label}</span>
                    <div style={{ fontSize: '0.85rem', color: '#64748b', lineHeight: '1.4' }}>{item.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
        <button className="save-btn" onClick={()=>onSave(style, depth)} style={{ marginTop: '2.5rem', width: '100%', padding: '1rem', borderRadius: '0.75rem', backgroundColor: '#DA7756', color: 'white', fontWeight: 'bold', fontSize: '1.1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', border: 'none', cursor: 'pointer', boxShadow: '0 4px 6px -1px rgba(218, 119, 86, 0.2)' }}>
          <Check size={20}/> Initialize Research Agent
        </button>
      </div>
    </div>
  );
};

function App() {
  const [sessionId] = useState(uuidv4());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  
  const [showArtifacts, setShowArtifacts] = useState(false); // To control the panel visibility
  const [artifacts, setArtifacts] = useState([]); // To store generated artifacts
  const [activeArtifact, setActiveArtifact] = useState(null); // To show in the modal
  
  const [showModal, setShowModal] = useState(false);
  const [prefs, setPrefs] = useState({ style: null, depth: null });
  const [pendingMessage, setPendingMessage] = useState(null); // STORES MESSAGE FOR RETRY
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isLoading]);

  const handleSend = async (manualInput = null, prefsOverride = null) => {
    const textToSend = manualInput || input;
    
    // ENFORCED: Must have text. 
    // Button is disabled otherwise, but this catches 'Enter' key on empty input.
    if (!textToSend.trim()) return;

    if (!hasStarted) setHasStarted(true);

    // Only add to UI if it's a new message (not a retry)
    if (!manualInput) {
      setMessages(prev => [...prev, { role: "user", content: textToSend }]);
      setInput("");
    }
    
    setIsLoading(true);

    // Use overridden prefs if provided (for immediate retry), otherwise state
    const currentStyle = prefsOverride?.style || prefs.style;
    const currentDepth = prefsOverride?.depth || prefs.depth;

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: textToSend, style_pref: currentStyle, depth_pref: currentDepth }),
      });
      const data = await res.json();
      
      // --- INTERCEPTOR ---
      if (data.response === "SURVEY_REQUIRED") {
        setPendingMessage(textToSend); // Save "Explain X"
        setIsLoading(false);
        setShowModal(true); // Open Survey
        return; 
      }

      setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
      if (data.artifacts && data.artifacts.length > 0) { // Check for artifacts
        setArtifacts(data.artifacts);
        setShowArtifacts(true); // Show the panel
      }
    } catch (e) { setMessages(prev => [...prev, { role: "system", content: "Error connecting to Agent." }]); }
    finally { setIsLoading(false); }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData(); formData.append("file", file);
    try {
      const res = await fetch(`${API_URL}/upload?session_id=${sessionId}`, { method: "POST", body: formData });
      if (res.ok) {
        if (!hasStarted) setHasStarted(true);
        // Just notify user. DO NOT open modal. User must type prompt.
        setMessages(prev => [...prev, { role: "system", content: `📎 Attached: ${file.name}` }]);
      }
    } catch (error) { console.error("Upload failed", error); }
  };

  const handleSurveySave = (style, depth) => {
    const newPrefs = { style, depth };
    setPrefs(newPrefs);
    setShowModal(false);
    
    // Resume the paused message
    if (pendingMessage) {
      handleSend(pendingMessage, newPrefs); 
      setPendingMessage(null);
    }
  };

  return (
    <div className={`app-container ${hasStarted ? 'active' : 'initial'}`}>
      <SurveyModal isOpen={showModal} onClose={()=>setShowModal(false)} onSave={handleSurveySave}/>
      <ArtifactViewer artifact={activeArtifact} onClose={()=>setActiveArtifact(null)} />
      <input type="file" ref={fileInputRef} onChange={handleFileUpload} style={{display:'none'}}/>

      <div className="hero-section" style={{
        position: 'absolute',
        top: '35%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 20,
        opacity: hasStarted ? 0 : 1,
        pointerEvents: hasStarted ? 'none' : 'auto',
        transition: 'opacity 0.4s ease-out',
        width: '100%'
      }}>
        <div className="hero-logo" style={{ marginBottom: '1.5rem' }}><SwishPlane size={80}/></div>
        <h1 style={{ fontSize: '3rem', fontWeight: '800', marginBottom: '0.5rem', color: '#1e293b', letterSpacing: '-0.02em' }}>PaperEngine.ai</h1>
        <p style={{ fontSize: '1.25rem', color: '#64748b' }}>Your AI Research Companion</p>
      </div>

      <div className="workspace" style={{ position: 'relative' }}>
        <FloatingBackground />
        <div className={`chat-panel ${showArtifacts ? 'shrunk' : 'full'}`} style={{
          zIndex: 10, 
          backgroundColor: hasStarted ? '#ffffff' : 'transparent', 
          transition: 'background-color 0.8s cubic-bezier(0.16, 1, 0.3, 1)', 
          position: 'relative', 
          boxShadow: 'none',
          display: 'flex',
          flexDirection: 'column',
          height: '100%'
        }}>
             <div className="chat-content" style={{ flex: 1, overflowY: 'auto', paddingBottom: '1rem' }}>
               {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: '1rem', padding: '0 1rem' }}>
                  {msg.role === "system" ? <div className="system-log" style={{ margin: '0 auto', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#94a3b8', fontSize: '0.85rem', backgroundColor: '#f1f5f9', padding: '0.5rem 1rem', borderRadius: '2rem' }}><FileText size={14}/><span>{msg.content}</span></div> 
                  : <div className="message-content" style={{ display: 'flex', maxWidth: '85%', gap: '0.75rem', alignItems: 'flex-start', flexDirection: msg.role === 'user' ? 'row' : 'row' }}>
                    {msg.role==='assistant'&&<div style={{flexShrink: 0, width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#FFF0EB', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '0'}}><SwishPlane size={20}/></div>}
                    <div className="markdown-body" style={{
                      backgroundColor: msg.role === 'user' ? '#DA7756' : '#F8FAFC',
                      color: msg.role === 'user' ? '#FFFFFF' : '#334155',
                      padding: '0.75rem 1.25rem',
                      borderRadius: '1rem',
                      borderTopLeftRadius: msg.role === 'assistant' ? '0' : '1rem',
                      borderTopRightRadius: msg.role === 'user' ? '0' : '1rem',
                      boxShadow: 'none',
                      fontSize: '0.95rem',
                      lineHeight: '1.5',
                      border: msg.role === 'assistant' ? '1px solid #e2e8f0' : 'none'
                    }}>
                      <ReactMarkdown components={{
                        p: ({node, ...props}) => <p style={{ margin: 0 }} {...props} />,
                        a: ({node, ...props}) => <a style={{ color: msg.role === 'user' ? '#fff' : '#DA7756', textDecoration: 'underline' }} {...props} />,
                        ul: ({node, ...props}) => <ul style={{ margin: '0 0 1rem 1.5rem', listStyleType: 'disc' }} {...props} />,
                        ol: ({node, ...props}) => <ol style={{ margin: '0 0 1rem 1.5rem', listStyleType: 'decimal' }} {...props} />,
                        li: ({node, ...props}) => <li style={{ marginBottom: '0.25rem' }} {...props} />,
                        h1: ({node, ...props}) => <h1 style={{ fontSize: '1.5em', fontWeight: 'bold', margin: '1rem 0 0.5rem 0' }} {...props} />,
                        h2: ({node, ...props}) => <h2 style={{ fontSize: '1.3em', fontWeight: 'bold', margin: '1rem 0 0.5rem 0' }} {...props} />,
                        h3: ({node, ...props}) => <h3 style={{ fontSize: '1.1em', fontWeight: 'bold', margin: '1rem 0 0.5rem 0' }} {...props} />,
                        code: ({node, inline, className, children, ...props}) => !inline ? 
                          <div style={{ backgroundColor: msg.role === 'user' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)', padding: '1rem', borderRadius: '0.5rem', overflowX: 'auto', margin: '1rem 0' }}><code className={className} style={{ fontFamily: 'monospace', fontSize: '0.9em' }} {...props}>{children}</code></div> : 
                          <code className={className} style={{ backgroundColor: msg.role === 'user' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)', padding: '0.2rem 0.4rem', borderRadius: '0.25rem', fontFamily: 'monospace', fontSize: '0.9em' }} {...props}>{children}</code>
                      }}>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.role==='user'&&<div style={{flexShrink: 0, width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '0'}}><User size={20} color="#64748b"/></div>}
                  </div>}
                </div>
              ))}

              {isLoading && <div className="message-row assistant"><div className="message-content"><div style={{marginTop:4}}><SwishPlane size={24}/></div><div style={{color:'#DA7756',fontStyle:'italic'}}>Thinking...</div></div></div>}
              <div ref={messagesEndRef}/>
             </div>

             {/* INPUT CONTAINER MOVED INSIDE CHAT PANEL FOR ALIGNMENT */}
             <div className="input-container" style={{
                position: hasStarted ? 'relative' : 'absolute',
                bottom: hasStarted ? 'auto' : 'auto',
                top: hasStarted ? 'auto' : '55%',
                left: hasStarted ? 'auto' : '50%',
                transform: hasStarted ? 'none' : 'translate(-50%, -50%)',
                width: '100%',
                display: 'flex',
                justifyContent: 'center',
                padding: '1.5rem',
                zIndex: 30,
                transition: 'all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1)',
                pointerEvents: 'none' // Container ignores clicks, box accepts them
              }}>
                <div className="input-box" style={{
                  width: '100%',
                  maxWidth: hasStarted ? '900px' : '650px',
                  backgroundColor: 'white',
                  borderRadius: '1rem',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
                  padding: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  border: '1px solid #e2e8f0',
                  pointerEvents: 'auto', // Re-enable clicks
                  transition: 'max-width 0.8s cubic-bezier(0.16, 1, 0.3, 1)'
                }}>
                  <button className="utility-btn" onClick={()=>fileInputRef.current.click()}><Plus size={20}/></button>
                  <button className="utility-btn" onClick={()=>setShowModal(true)}><SlidersHorizontal size={18}/></button>
                  <input type="text" placeholder="Research query..." value={input} onChange={(e)=>setInput(e.target.value)} onKeyDown={(e)=>e.key==='Enter'&&handleSend()} style={{ flex: 1, border: 'none', outline: 'none', fontSize: '1rem', background: 'transparent', color: '#1e293b' }}/>
                  <button className="send-btn" onClick={()=>handleSend()} disabled={!input.trim()}><Send size={18}/></button>
                </div>
              </div>
        </div>

        <div className={`artifact-panel ${showArtifacts ? 'visible' : 'hidden'}`} style={{ backgroundColor: 'rgba(249, 250, 251, 0.85)', backdropFilter: 'blur(12px)', borderLeft: '1px solid rgba(0,0,0,0.05)' }}>
          <div className="artifact-header">
            <h3>Generated Materials</h3>
            <button onClick={()=>setShowArtifacts(false)} className="close-panel"><X size={20}/></button>
          </div>
          <div className="artifact-content">
            {artifacts.map((art, i) => (
              <div key={i} className="artifact-card">
                <div className="card-header"><span className="badge">{art.type}</span><Copy size={14} className="copy-icon"/></div>
                <div className="card-body">
                   {art.type.includes("Quiz") ? `${art.data.quiz?.length || 5} Questions` : <ReactMarkdown>{typeof art.data === 'string' ? art.data.substring(0, 300) + "..." : "Click to view content"}</ReactMarkdown>}
                </div>
                <button className="view-btn" onClick={()=>setActiveArtifact(art)}>View Full {art.type}</button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;