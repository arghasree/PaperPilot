"""
I/O helpers for LaTeX files: read, paragraph splitting, and write-back.
"""

import re
from pathlib import Path

_BEGIN = re.compile(r'\\begin\{[^}]+\}')
_END = re.compile(r'\\end\{[^}]+\}')


def read_tex(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_tex(path: str | Path, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def split_paragraphs(text: str) -> list[str]:
    """
    Split LaTeX text into paragraphs at blank lines or \\par.

    Blank lines inside \\begin{}...\\end{} environments are not treated as
    paragraph breaks, so equation/align/itemize blocks stay attached to their
    surrounding prose.
    """
    # Normalise \par to a blank line so we only have one case to handle
    text = re.sub(r'\\par\b', '\n', text)

    depth = 0
    current: list[str] = []
    paragraphs: list[str] = []

    for line in text.splitlines():
        depth += len(_BEGIN.findall(line))
        depth -= len(_END.findall(line))
        depth = max(depth, 0)  # guard against malformed LaTeX

        if not line.strip() and depth == 0:
            chunk = "\n".join(current).strip()
            if chunk:
                paragraphs.append(chunk)
            current = []
        else:
            current.append(line)

    chunk = "\n".join(current).strip()
    if chunk:
        paragraphs.append(chunk)

    return paragraphs


def join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def extract_body(text: str) -> tuple[str, str, str]:
    """
    For a full .tex document, return (preamble, body, postamble).

    preamble  — everything up to and including \\begin{document}
    body      — the content between \\begin{document} and \\end{document}
    postamble — \\end{document} and anything after

    If the document markers are absent (e.g. input is just a section),
    returns ("", text, "").
    """
    begin_marker = r'\begin{document}'
    end_marker = r'\end{document}'
    begin_pos = text.find(begin_marker)
    end_pos = text.find(end_marker)

    if begin_pos == -1 or end_pos == -1:
        return "", text, ""

    preamble = text[: begin_pos + len(begin_marker)]
    body = text[begin_pos + len(begin_marker) : end_pos]
    postamble = text[end_pos:]
    return preamble, body, postamble


"""
  Single paragraph mode (input is one paragraph in a .tex file):  
                        
  from utils.latex_io import read_tex, write_tex
                                                                                        
  text = read_tex("para.tex")                                                           
  # text is just the raw string — pass it directly to agents                            
  refined = rewriter.run(text, tech_critique, writing_critique)                         
  write_tex("output/output.tex", refined)                                               
                                                
 ------------------                                                                 
  Multi-paragraph mode (input is a full section):    
                                     
  from utils.latex_io import read_tex, write_tex, split_paragraphs, join_paragraphs     
                                                                                        
  text = read_tex("section_method.tex")                                                 
  paragraphs = split_paragraphs(text)                                                   
  # e.g. paragraphs = ["We propose...", "The model consists of...", "Training           
  proceeds..."]                                                                         
                                                                                        
  refined_paragraphs = []                                                               
  for para in paragraphs:                                                               
      tech   = technical_critic.run(para, topic)                                        
      writing = writing_critic.run(para, section_type)                                  
      refined = rewriter.run(para, tech, writing)                                       
      refined_paragraphs.append(refined)                                                
                                                                                        
  # Reviewer gets the full section for flow check                                       
  final_text = join_paragraphs(refined_paragraphs)                                      
  write_tex("output/output.tex", final_text)                                            
                                                               
 ------------------                                                                                       
  Full document input (has \documentclass, preamble, etc.):      
                         
  from utils.latex_io import read_tex, write_tex, extract_body, split_paragraphs,       
  join_paragraphs                                                                
                                                                                        
  text = read_tex("paper.tex")
  preamble, body, postamble = extract_body(text)                                        
                                                
  paragraphs = split_paragraphs(body)                                                   
  refined_paragraphs = [...]  # same loop as above
                                                  
  write_tex("output/output.tex", preamble + join_paragraphs(refined_paragraphs) +       
  postamble) 
"""