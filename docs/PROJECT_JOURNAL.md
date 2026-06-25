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

---

## 2026-06-25 — B8 sweep block-dim air: nessuna config batte 32x2x2 (finding negativo)
**Contesto:** sweep geometria blocco air (cart+fcc) per ridurre il tempo del
kernel dominante (~36% Nsight).
**Metodo:** 6 config (32x2x2 base, 32x4x2, 32x8x1, 64x2x2, 64x4x1, 128x2x1),
5 run/config, poi interleaved 10-round anti-clock-drift su large grid.
Gate correttezza: tutte bit-identiche (block-shape non tocca i valori).
**Risultato:** sweep iniziale dava 32x8x1 +5.7% sul min, MA era jitter di clock
(GB10, no clock-lock senza root). Interleaved: 32x8x1 = +1.8% min, PEGGIORE su
median/mean. launch_bounds peggiora (limita occupancy, zero beneficio banda).
Nessuna config batte 32x2x2 >5% in modo robusto.
**Causa fisica (HW counters non disponibili - ERR_NVGPUCTRPERM, serve root;
evidenza sostitutiva misurata):**
- CUPTI kernel-trace (no HW counter), large grid, 1020 istanze KernelAirCart:
  32x2x2 = 8.701 ms avg / 8.607 min; 32x8x1 = 8.625 / 8.514. Delta ~0.9% avg
  (~1.1% min), DENTRO lo StdDev (~1.6%). Il kernel air dominante (~57% del tempo)
  e' insensibile alla forma del blocco -> la geometria di lancio non e' la leva.
- Banda effettiva (STIMA analitica, lower-bound, non misura HW): 7-punti su 159.9M
  celle in ~8.70 ms, traffico DRAM minimo con riuso L2 ideale = 3*Npts*4B
  ~1.92 GB/step -> ~220 GB/s ~= 80% del picco GB10 (~273 GB/s LPDDR5X). Essendo
  lower-bound (riuso reale imperfetto), la saturazione effettiva e' >=. Coerente
  con kernel bandwidth-bound, banda ~satura.
- dram__throughput HW non ottenibile in questo ambiente (counter ristretti).
  La causa e' confermata INDIRETTAMENTE (kernel-time insensibile + banda stimata
  satura), non via counter diretto.
**Decisione:** default 32x2x2 invariato, nessun codice. Non ritentare il
block-dim su questa GPU: la leva non e' la geometria di lancio ma la banda
(-> e' li' che punta il rewrite z-slab/tiling B9, se serve).

---

## 2026-06-25 — B9 z-slab tiling air: -14% su GB10, APERTO su RTX 4500 Ada
**GB10 (LPDDR5X unified, sm_121):** tiled vs ref interleaved 10-round, CUPTI.
ref 8.726 ms min, tiled 10.116 ms min -> -13.7% (median -13.9%, StdDev 0.15%,
robusto, no jitter). Banda stimata 220->190 GB/s. Correttezza Delta=0.
**Causa:** z-slab march riduce il parallelismo (160M thread -> (Nz-2)(Ny-2)
thread con loop seriale su cz); su kernel memory-bound il parallelismo che
nasconde la latenza conta piu' del register-reuse, che la L2 GB10 (grande,
cattura le 3 slab ~2.1MB) dava gia' gratis.
**APERTO su RTX 4500 Ada:** architettura memoria diversa (GDDR6 dedicata,
L2 piu' piccola). Se la L2 Ada NON cattura il working-set 3-slab, il bilancio
parallelismo-vs-reuse puo' girare. Codice in branch wip/b9-air-tiled, bench
Ada da eseguire sulla workstation. NON archiviato come negativo definitivo.
