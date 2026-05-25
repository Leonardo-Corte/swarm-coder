# SwarmCoder v5 — Task Plan: 4 Intelligence Upgrades
_Obiettivo: rendere SwarmCoder più intelligente di Claude Code su task di coding_
_Data: 2026-05-25_

---

## Problema da risolvere

Il modello locale (14B/16B parametri) ha 4 debolezze specifiche:

| # | Problema | Impatto |
|---|---|---|
| 1 | Multi-file senza planning → interfacce inconsistenti | ⚠️ |
| 2 | Architettura complessa → risposte generiche | ⚠️ |
| 3 | Refactoring repo grande → rompe altri file | ❌ |
| 4 | Debugging oscuro → loop infiniti | ❌ |

---

## FASE 1 — Architect Mode (Problema 1)
_Pattern: Aider architect_coder.py — pianifica prima, implementa dopo_

### Step 1.1 — Leggi e studia architect_coder.py di Aider
- File: `/Users/corte/swarm-coder/aider/aider/coders/architect_coder.py`
- Estrai: il prompt esatto, il pattern two-pass, come passa il piano all'editor

### Step 1.2 — Aggiungi `ARCHITECT_SYSTEM` prompt a main.py
- Prompt che forza il modello a produrre SOLO la struttura prima del codice:
  - Lista file da creare con path
  - Interfacce tra moduli (import, function signatures)
  - Ordine di implementazione
- Il prompt deve VIETARE di scrivere codice in questa fase

### Step 1.3 — Aggiungi tool `plan_project(task, directory)`
- Invoca il modello con ARCHITECT_SYSTEM
- Salva il piano in `_NOTES["architect_plan"]`
- Ritorna la struttura file + interfacce in formato leggibile

### Step 1.4 — Aggiungi tool `implement_plan(plan)`
- Prende il piano da `_NOTES["architect_plan"]`
- Implementa file per file in ordine
- Dopo ogni file: verifica import e sintassi con `run_python`

### Step 1.5 — Aggiorna system prompt orchestratore
- Aggiungi regola: "Per task multi-file: chiama SEMPRE plan_project() prima di scrivere codice"
- Aggiungi esempi nel formato tool call

### Step 1.6 — Test Architect Mode
- Task test: "crea un tool Python CLI con 4 moduli: parser, stats, report, main"
- Verifica: il modello pianifica prima, poi implementa in ordine
- Verifica: tutti gli import funzionano al primo run

### Step 1.7 — Commit Fase 1
```bash
git commit -m "feat: v5 architect mode — two-pass planning before implementation"
```

---

## FASE 2 — Hypothesis Debugger (Problema 4)
_Pattern: InspectCoder — ipotesi → test → analisi, con loop detection_

### Step 2.1 — Studia SWE-agent debugging patterns
- Directory: `/Users/corte/swarm-coder/SWE-agent/`
- Cerca: come struttura il debugging loop, come evita loop infiniti

### Step 2.2 — Aggiungi `HypothesisTracker` class
```python
class HypothesisTracker:
    def __init__(self):
        self.hypotheses: list[str] = []
        self.test_results: list[str] = []
        self.iteration: int = 0

    def is_looping(self) -> bool:
        # Se ultimi 2 risultati identici → loop
        ...

    def add(self, hypothesis: str, result: str): ...
    def summary(self) -> str: ...
```

### Step 2.3 — Aggiungi tool `debug_hypothesis(error, hypothesis, test_command)`
- Riceve: errore da debuggare, ipotesi del modello, comando di test
- Esegue il test, cattura output
- Compara con ipotesi: confermata / refutata / inconclusiva
- Aggiorna HypothesisTracker
- Se loop detected: ritorna "STUCK — cambia approccio" con suggerimento

### Step 2.4 — Aggiungi tool `start_debug_session(error_description)`
- Inizializza HypothesisTracker fresco
- Prompt speciale al modello: "Formula ipotesi FALSIFICABILE su causa dell'errore"
- MAX 5 iterazioni per sessione

### Step 2.5 — Loop detection nel ReAct loop principale
- Se il modello chiama lo stesso tool con gli stessi argomenti 3 volte → interrompi
- Inietta nel contesto: "Stai ripetendo la stessa azione. Cambia strategia."

### Step 2.6 — Test Hypothesis Debugger
- Crea un file con bug intenzionale
- Verifica che il debugger: formula ipotesi, testa, non va in loop
- Verifica che risolve entro 5 iterazioni

### Step 2.7 — Commit Fase 2
```bash
git commit -m "feat: v5 hypothesis debugger — structured debug loop with loop detection"
```

---

## FASE 3 — Constraint-Aware Architecture (Problema 2)
_Pattern: DSPy Assertions — forza il modello a ragionare su vincoli prima di rispondere_

### Step 3.1 — Aggiungi `CONSTRAINT_EXTRACTION_PROMPT`
- Forza il modello a estrarre strutturatamente:
  - Vincoli hard (budget, latenza, dipendenze max)
  - Vincoli soft (preferenze)
  - Trade-off da considerare

### Step 3.2 — Aggiungi tool `extract_constraints(problem_description)`
- Chiama il modello con CONSTRAINT_EXTRACTION_PROMPT
- Ritorna JSON strutturato: `{hard: [...], soft: [...], tradeoffs: [...]}`

### Step 3.3 — Aggiungi tool `compare_architectures(options, constraints)`
- Prende 2-3 opzioni architetturali
- Per ognuna: score su performance, manutenibilità, costo, complessità
- Ritorna matrice comparativa

### Step 3.4 — Integra nel system prompt
- "Per domande architetturali: PRIMA extract_constraints(), POI compare_architectures()"

### Step 3.5 — Commit Fase 3
```bash
git commit -m "feat: v5 constraint reasoning — structured architecture decisions"
```

---

## FASE 4 — Dependency Graph (Problema 3)
_Pattern: CodeGraph (NetworkX + AST) — sa cosa rompe prima di cambiare_

### Step 4.1 — Implementa `build_dependency_graph(directory)`
- Parsa tutti i file Python con `ast`
- Costruisce grafo NetworkX: nodo = file/funzione, arco = import/chiamata
- Salva il grafo in memoria (o su disco come JSON)

### Step 4.2 — Aggiungi tool `find_usages(symbol, directory)`
- Dato un nome funzione/classe: trova tutti i file che la usano
- Usa il grafo o grep strutturato
- Ritorna: lista file + linee

### Step 4.3 — Aggiungi tool `impact_analysis(file_path, changes)`
- Prima di modificare un file: calcola impatto
- Ritorna: lista file che potrebbero rompersi + severity (high/medium/low)

### Step 4.4 — Aggiungi tool `safe_refactor(old_symbol, new_symbol, directory)`
- Trova tutti gli usi con find_usages()
- Aggiorna tutti i file in modo coordinato
- Verifica import dopo ogni modifica

### Step 4.5 — Integra nel system prompt
- "Prima di modificare una funzione/classe pubblica: chiama SEMPRE impact_analysis()"

### Step 4.6 — Test su repo reale
- Prova a rinominare una funzione usata in 3+ file
- Verifica che safe_refactor() aggiorna tutti i file correttamente

### Step 4.7 — Commit Fase 4
```bash
git commit -m "feat: v5 dependency graph — safe refactoring with impact analysis"
```

---

## FASE 5 — Integrazione finale e push

### Step 5.1 — Aggiorna banner a v5
### Step 5.2 — Aggiorna README con nuove capacità
### Step 5.3 — Aggiorna requirements.txt (networkx)
### Step 5.4 — Test integrazione completa
### Step 5.5 — Push su GitHub
```bash
git push origin main
```

---

## Stato avanzamento

- [x] Fase 1: Architect Mode
- [x] Fase 2: Hypothesis Debugger
- [x] Fase 3: Constraint Reasoning
- [x] Fase 4: Dependency Graph
- [x] Fase 5: Integrazione finale
