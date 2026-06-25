# PROJECT JOURNAL — pffdtd (fork)

Record append-only delle modifiche al fork. Storia autorevole: mai riscrivere
il passato, solo aggiungere in coda. Ogni sprint/fix/audit ha una entry datata,
aggiornata nello STESSO commit della modifica (non dopo).

Formato per entry:

## YYYY-MM-DD — <titolo sprint>
**Contesto:** perché si interviene
**Modifiche:** cosa, con hash commit e file/righe
**Razionale:** perché questa scelta e non altre
**Esito:** gate, PASS/FAIL, stato
**Aperto:** cosa resta

---

## 2026-06-25 — Lotto A-safe: ottimizzazione GPU forward
**Contesto:** kernel FDTD memory-bound; audit ha individuato stalli per-timestep
e config build obsoleta (sm_35 PTX-JIT). Matematica del kernel invariata.
**Modifiche:**
- 5377f03 A1: Makefile -arch=native + -Xptxas -O3 -lineinfo (era sm_35); +2/-1
- 4914dd8 A2: P2P cudaDeviceEnablePeerAccess + fallback host-staging; +19
- 49a8aee A3: AddInBatch, in_sigs grezzo su device, negazione solo nel kernel,
  cuBrw, n int64_t (+24/-3); elimina Ns×Nt micro-lanci <<<1,1>>>
- 2fed42c A5: readout a blocchi READOUT_BLOCK=512, layout [col*Nr+nr] (+25/-18);
  elimina memcpy+sync per-step
**Razionale:** A1/A2 bit-identici; A3/A5 cambiano ordine/timing non valori
(celle distinte, somma bit-identica). Tutti i diff verificati con git diff -w
senza reformat.
**Esito:** 4 commit puliti su develop (A1, A2, A3, A5), HEAD=2fed42c.
develop locale era 2 commit dietro origin/develop (A3/A5 erano solo sul remoto,
GB10 indietro rispetto al push dal Mac); allineato con fast-forward pulito
(0 commit locali) prima di documentare. Gate build+numerico (sim_outs.h5 vs
baseline a machine accuracy) DA ESEGUIRE sul GB10 con CUDA+GPU.
**Aperto:** validazione numerica + benchmark (baseline pre-A1 vs HEAD); poi
decisione A6 (u1b, gated) vs Lotto B. A4 (progress/sync ogni N step): NON
committato — non presente in alcun ref del repo; resta da fare (o scartato),
quindi NON figura tra le Modifiche. NB: il paragrafo generico del README cita
"progress/sync" tra le aree di lavoro CUDA, ma nessun commit lo implementa: da
riconciliare quando/se A4 sarà fatto.

---

## 2026-06-25 — A6 (rimozione u1b): archiviato come finding NEGATIVO

**Contesto:** l'audit segnalava u1b come "buffer ridondante" (KernelBoundaryFD
legge solo u0b/u2b). Ipotesi: eliminarlo collassando la rotazione boundary a 2 slot
per risparmiare VRAM.

**Analisi (per induzione sulla griglia):** B(n-1) e' usato due volte per step —
da KernelBoundaryRigidCart (termine -u0[bnl], poi sovrascritto) e da KernelBoundaryFD
(come u2b, correzione impedenza). u2b esiste proprio per preservare B(n-1) dopo che
RigidCart ha sovrascritto la copia in griglia. All'inizio dello step n, u0[bnl]
contiene esattamente B(n-1) (scritto da CopyToGrid a fine n-1, mai toccato da air
[mask] ne' RigidCart [non scrive u1]).

**Eliminazione SAFE esiste:** rimuovere u1b e ripopolare u2b ogni step con
CopyFromGrid(u2b <- u0[bnl]) prima di RigidCart. Numericamente identica -> gate A6
darebbe Delta=0.

**Perche' NON si fa (trade perdente):**
- Risparmio: Nbl*sizeof(Real) ~15 MB/GPU = <0.1% su 24-128 GB.
- Costo: +1 gather CopyFromGrid/step sul path boundary HOT (~31 MB traffico/step
  x 7646 step sulla repr), sostituisce una rotazione a puntatori che oggi e' gratis.
  Su kernel memory-bound = rallentamento netto, opposto dell'obiettivo del lotto.
- Fragilita' multi-GPU: l'equivalenza u0[bnl]==B(n-1) e' pulita su 1 GPU; con
  halo/comms multi-GPU andrebbe verificata nodo per nodo.

**Conclusione:** u1b e' il meccanismo corretto e piu' economico (carry a puntatore,
costo zero). A6 archiviato, nessun codice. Non ritentare: l'audit pesava la memoria
ma non il costo del gather sostitutivo.
