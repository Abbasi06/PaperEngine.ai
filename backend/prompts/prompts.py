# prompts.py

# 1. COMPREHENSIVE ENTITY EXTRACTION (For KG & Research)
# We ask for ALL significant terms to populate the Knowledge Graph nodes.
KEYWORD_PROMPT = """
Analyze the following text and extract ALL significant technical entities, concepts, and terminologies.

Target Output:
- Focus on nouns and noun phrases that could act as 'Nodes' in a Knowledge Graph.
- Include: Specific algorithms, metrics, researchers, tool names, historical events, and core concepts.
- Exclude: Generic verbs, stopwords, or vague adjectives.

Format: Return ONLY a JSON object with a single list.
Example:
{{
  "entities": ["Transformer Architecture", "Attention Mechanism", "Vaswani et al.", "BLEU Score", "NLP"]
}}

Text: {text}
"""

# 2. DYNAMIC QUIZ (Unchanged but included for context)
QUIZ_PROMPT = """
You are a professor creating a generic quiz.
Context: {context}

Target Audience: {depth} (e.g., Layman = simple words, Deep = technical).
Learning Style: {style} (e.g., Visual = ask about charts/diagrams).

Task: Generate 5 multiple-choice questions.
Format: Return a JSON object with a key "quiz" containing a list of questions.
Example: {{ "quiz": [ {{ "question": "...", "options": ["..."], "answer": "...", "explanation": "..." }} ] }}
"""

# 3. DYNAMIC FLASHCARDS
FLASHCARD_PROMPT = """
Create 5 flashcards based on this context: {context}
Complexity Level: {depth}.

If the user style is 'Visual', try to describe concepts that can be visualized.
Format: Return a JSON object with a key "flashcards" containing a list.
Example: {{ "flashcards": [ {{ "front": "...", "back": "..." }} ] }}
"""

# 4. DYNAMIC SUMMARY
SUMMARY_PROMPT = """
Summarize the following text: {context}

Constraint 1 (Depth): {depth} 
- If 'Layman': Use analogies, avoid jargon.
- If 'Technical': Use precise terminology and math.

Constraint 2 (Style): {style}
- If 'Visual': Describe the mental image of the concept.
- If 'Text': Focus on structure and flow.

Return 3 paragraphs.
"""

# 5. MINDMAP (Mermaid)
MINDMAP_PROMPT = """
Create a Mermaid.js mindmap for: {context}
Complexity: {depth}.
Return ONLY the code starting with `graph TD`.
"""

# 6. PRESENTATION OUTLINE (PPT)
PPT_PROMPT = """
Create a structured presentation outline for: {context}
Audience Level: {depth}.

Format: Return a JSON object with a key "slides".
Each slide must have: "title", "bullets" (list of strings), and "speaker_notes".
Example:
{{
  "slides": [
    {{ "title": "Introduction", "bullets": ["Point 1", "Point 2"], "speaker_notes": "Say this..." }}
  ]
}}
"""

# 7. VIDEO SCRIPT
VIDEO_SCRIPT_PROMPT = """
Write a YouTube video script based on: {context}
Style: {style} (e.g., Educational, Fast-paced).
Depth: {depth}.

Format: Return a JSON object with a key "script" containing a list of sections.
Each section: {{ "timestamp": "0:00", "visual": "...", "audio": "..." }}
"""

# 8. PODCAST SCRIPT
PODCAST_SCRIPT_PROMPT = """
Write a Podcast script (Host & Guest format) based on: {context}
Depth: {depth}.

Format: Return a JSON object with a key "script" containing a list of dialogue turns.
Each turn: {{ "speaker": "Host" or "Guest", "text": "..." }}
"""
