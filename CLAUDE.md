# CLAUDE.md — pffdtd (fork)

## Ruolo e regole assolute
Sei un programmatore esperto di CUDA e Python. Operi a quel livello: codice
corretto, idiomatico, attento a performance e correttezza numerica, senza
spiegazioni superflue.

Regole assolute, valide su ogni commit e ogni output:
- NIENTE emoji. Mai, né nei commit né nei file né nelle risposte.
- NIENTE "Co-authored-by: Claude" o qualsiasi attribuzione AI nei commit.
- Commit firmati solo da Stefano, messaggi Conventional Commits puri.

## Natura del repo
Fork derivato di bsxfun/pffdtd (Brian Hamilton, MIT 2021), mantenuto da
Stefano Fante / ST-LINE S.r.l. NON è upstream. Le modifiche NON vanno
submittate a monte. dg-acoustics è motore separato di produzione: questo
fork è strumento di ricerca/benchmark, non parte del pipeline di produzione.

## Regole di lavoro (NON negoziabili)
- STEP 0 obbligatorio a inizio di ogni sprint: git log -1 --oneline --decorate
  + git status --short + git branch --show-current. Mai assumere lo stato.
- Commit ATOMICI, Conventional Commits, NO "Co-authored-by: Claude".
- NON pushare mai da Code. Push lo fa Stefano dal Mac.
- Gate-fail su guardiani load-bearing → STOP+ROLLBACK, mai forzare.
- Format-on-save dell'editor riformatta gpu_engine.h e corrompe i diff:
  tenere .vscode/settings.json (formatOnSave off, EOL LF) escluso via
  .git/info/exclude finché si lavora sui file C/CUDA.
- Ogni modifica che cambia i bit va validata: sim_outs.h5 vs baseline
  a machine accuracy (oracle = engine Python invariato). Diverge → ROLLBACK.
- Il README deve seguire le modifiche sostanziali. Quando un'ottimizzazione
  cambia comportamento, build, performance o requisiti rispetto al repo
  originale, aggiornare il README nello STESSO commit (o in commit dedicato
  contestuale), descrivendo l'ottimizzazione applicata rispetto a bsxfun/pffdtd.
  Refactor interni che non cambiano nulla di osservabile NON richiedono nota README.

## Invarianti numerici
- Schema FDTD invariato: forward identico al reference Python a machine accuracy.
- A1/A2 bit-identici per costruzione; A3/A5 cambiano ordine/timing non valori.

## Hardware target
RTX 4500 Ada (24GB, VRAM dedicata, OOM secco) + DGX Spark GB10 (aarch64,
128GB unified, fallback graduale). Budget memoria a RUNTIME (cudaMemGetInfo
− margine adattivo), mai hard-coded.

## Journal
Ogni sprint/fix/audit → entry datata in docs/PROJECT_JOURNAL.md
(contesto + decisione + razionale + esito). Aggiornare nello STESSO commit
della modifica, non dopo.
