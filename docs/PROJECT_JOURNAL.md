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

---

## 2026-06-25 — B10 bn_mask: non rimovibile + guadagno <1% (finding negativo da analisi)
**Ipotesi audit:** il load bn_mask[ii>>3] per cella nel kernel air e' ridondante
se il boundary riscrive quei nodi DOPO l'air -> rimuoverlo toglierebbe uno stream
di load su 160M celle/step.
**Sequenza reale (verificata, gpu_engine.h:1076-1134):** l'ordine e' boundary -> air,
NON air -> boundary. cuStream_bn esegue RigidCart + CopyFromGrid + BoundaryFD
(impedenza) + CopyToGrid sui nodi bnl; poi cudaStreamWaitEvent serializza
cuStream_air DIETRO il boundary; l'air gira dopo. L'unico kernel post-air e'
KernelBoundaryABC (nodi bna = bordo assorbente, NON i bnl materiali).
**Kill-criterion 1 (load-bearing):** togliere il mask fa scrivere all'air u0[ii]
sui nodi bnl, sovrascrivendo il valore d'impedenza appena calcolato. Nessun kernel
rigira dopo a correggerlo -> corruzione deterministica. Il mask NON e' ridondante:
e' il meccanismo che protegge i nodi boundary dal clobber dell'air.
**Kill-criterion 2 (sotto soglia):** anche fosse safe, bn_mask = 1 byte/8 celle =
ceil(Npts/8) = 19.99 MB/step vs floor air 1.918 GB/step = 1.04%, per giunta
L2-resident -> impatto DRAM reale <1%, sotto la soglia 5%.
**Conclusione:** non rimovibile e comunque trascurabile. Archiviato, nessun codice.
L'audit aveva l'ordine degli stream invertito.

---

## 2026-06-25 — B11 halo flip fusion: 3 facce sempre-presenti in 1 lancio (committato)
**Contesto:** 6 kernel FlipHalo (ghost-cell mirror Neumann) lanciati separatamente
ogni step. Le 3 sempre-presenti (XZ_Ybeg, YZ_Xbeg, YZ_Xend) fuse in 1 kernel
FlipHaloFaces<Idx> (dispatch su blockIdx.z). Le 3 condizionali (XY_Zbeg/Zend gid,
XZ_Yend fcc) restano separate, lanciate PRIMA del fuso (ordine spigoli preservato).
Switch -DHALO_SEPARATE mantiene i 6 originali per A/B.
**Correttezza:** le facce NON sono disgiunte (il piano y=0 interseca le colonne
x=0/x=Nx-1 sugli spigoli); nei lanci separati YZ gira dopo XZ e vince. Nel fuso XZ
skippa quelle 2 colonne (di proprieta' YZ) -> writes disgiunte, stesso last-writer
-> Delta=0 (gate small cart E fcc).
**Bench (CUPTI flip-exec, interleaved 10-round, std ~0.5%):** large 824.2->825.3
us/step (neutro, -0.1%); small 80.57->79.22 us/step (+1.7% robusto, 1 rampa vs 3).
Combined loop-time troppo rumoroso (std 30% small / 6% large) per risolvere il
saving lanci.
**Esito:** guadagno reale ma piccolo (+1.7% sui flip small = ~0.1% del passo;
launch-overhead -2 lanci non misurabile). Committato perche' bit-identico,
marginalmente positivo e codice piu' pulito (6->4 lanci, 1-GPU cart). Nessun
effetto osservabile su build/comportamento -> nessuna nota README.

---

## 2026-06-25 — B12 packing vh1/gh1: padding fuori dal traffico (finding negativo da analisi)
**Ipotesi audit:** vh1/gh1 allocati Nbl x MMb (MMb=12) con padding -> traffico
sprecato su KernelBoundaryFD (15% del passo).
**Fatto decisivo (gpu_engine.h:419,431):** entrambi i loop sono for(m=0;m<cuMb[k];m++)
-> il kernel legge/scrive solo i cuMb poli USATI, NON i MMb allocati. Il padding e'
solo VRAM allocata, mai toccata dal loop. La premessa "padding -> traffico" e' falsa.
**cuMb reale (scena CTK):** tutti gli 8 materiali = 11 rami ADE su MMb=12. Padding
= 1 ramo (~8%), non 3x. Le impedenze sono fit ricchi a 11 poli.
**Traffico vh1/gh1:** ~91% del traffico DRAM per-nodo del kernel (176 B: read+write
11 poli x 2 campi x 4B), layout gia' pole-major [m*Nbl+nb] coalescato. E' traffico
REALE e irriducibile (11 poli usati), non padding.
**Guadagno atteso packing:** ~0% sul tempo (loop gia' minimale, legge solo cuMb;
padding gia' piccolo; layout gia' coalescato). Risparmio solo VRAM: ~31 MB su 128 GB.
**Verdetto:** padding fuori dal traffico + <2% -> archiviato, nessun codice.
Come B10, l'audit aveva il meccanismo invertito (pensava si leggesse MMb, si legge cuMb).

---

## 2026-10-09 — Audit CUDA: dipendenze halo e piano Ada/GB10
**Contesto:** riesame del forward e della pipeline completa per RTX 4500 Ada e
DGX Spark GB10, con calcolo numerico nativo C/C++/CUDA.
**Modifiche:** `docs/CUDA_AUDIT.md` documenta i finding sul commit `ad169d2`,
la dipendenza read-after-write tra facce nel kernel B11, le barriere host,
la pipeline boundary e il piano di migrazione del preprocessing/postprocessing.
**Razionale:** la disgiunzione delle scritture B11 non elimina la lettura YZ
da una cella aggiornata da XZ. Il modello host produce valori diversi per
due ordini consentiti ai blocchi; FCC legge i corner interessati.
**Esito:** PASS compilazione host FP32/FP64 e modello C delle dipendenze.
Nessuna simulazione Python eseguita. Analisi e prove host; nessun nuovo
benchmark GPU o gate RIR, in assenza di dispositivo/toolkit nella macchina.
**Aperto:** correggere B11, introdurre regressioni native, validare su GPU;
poi scheduler e boundary con gate numerici e benchmark separati per target.

---

## 2026-10-09 — Correzione halo B11 e fusione trasferimenti boundary
**Contesto:** l'audit sul commit `ad169d2` ha identificato una dipendenza RAW
tra le facce fuse, oltre a due lanci gather/scatter eliminabili dai bordi ADE.
**Modifiche:**
- `c_cuda/halo_faces.h` compone direttamente le letture degli spigoli; il
  kernel CUDA usa la stessa mappatura dei test e mantiene le facce ordinate
  per assi di tre celle. `HALO_SEPARATE` conserva il riferimento.
- `c_cuda/boundary_fd.h` e `KernelBoundaryFDGrid` fondono gather/ADE/scatter,
  conservando rigid stencil, ordine aritmetico, stati per polo e tre carry.
  `BOUNDARY_SEPARATE` conserva i tre passaggi originali.
- Test C/C++ nativi e CUDA in `tests/`; Makefile con `CUDA_ARCH`, `BUILD_DIR`
  e target `test-native`, `build-cuda-tests`, `test-cuda`.
- Enumerazione CUDA controllata prima della partizione; README allineato
  alla gestione memoria realmente implementata e ai nuovi comandi.
**Razionale:** le letture composte rimuovono la dipendenza fra blocchi senza
un lancio aggiuntivo. La fusione ADE rimuove due lanci e accessi espliciti
pari a 20 B/nodo/step FP32 o 32 B FP64; non sono misure del traffico DRAM.
**Esito:** PASS 640 casi halo host e, per precisione, 10 casi ADE x 37 passi,
incluse tutte le rotazioni, stati usati e padding; PASS sanitizer host.
PASS build CUDA FP32/FP64 per sm_89 e sm_121, percorsi fusi e separati;
PASS compilazione test CUDA. Esecuzione CUDA SKIP (exit 77): nessun device.
Nel build sm_89 FP32 i kernel ADE originale/fuso hanno entrambi 40 registri,
stack frame 96 B e zero spill dichiarati dal compilatore.
**Aperto:** gate numerico device, stabilità/RIR e benchmark su Ada e GB10;
sm_121 compilato su host x86_64 non sostituisce build host aarch64 DGX Spark.
Scheduler Graphs e pipeline prepare/postprocess nativa restano da implementare.

---

## 2026-10-09 — Scheduler a eventi: attese host per blocco di output
**Contesto:** il percorso a una GPU mantieneva tre attese stream e una attesa
evento di timing a ogni timestep, anche senza scambi peer.
**Modifiche:** `PFFDTD_ASYNC=1` abilita una catena di eventi a una GPU;
boundary(n) attende air(n-1), air(n) attende boundary(n), readout(n) precede
boundary(n+1) nello stesso stream. Drain e attesa host ogni 512 passi/fine.
Stream nonblocking con sincronizzazione iniziale dopo setup; puntatori bulk
e carry ruotano come prima. Ricevitori non interni e multi-GPU usano il
riferimento; `SCHEDULER_SYNC` lo forza. Telemetria async wall-time a blocchi,
`PFFDTD_PROGRESS=0` disabilita progressi. Corretto anche il progresso senza TTY
e la stima del tempo al passo zero in `fdtd_common.h`.
**Razionale:** le dipendenze device proteggono il riuso dei campi e del readout,
consentendo di accodare passi senza attendere l'host dopo ciascuno. Il percorso
rimane opzionale prima del gate hardware e non è ancora CUDA Graphs.
**Esito:** PASS contratto nativo su 576 DAG randomizzati, checkpoint int64,
rotazioni 2/3 e mutanti con dipendenze mancanti. Fixture CUDA completa
sync/async con 40 casi per precisione, Cart/FCC, ABC/ADE, sorgenti, ricevitori
duplicati e Nt attorno ai drain. PASS build CUDA FP32/FP64 e fixture per
sm_89 e sm_121, più build del riferimento forzato `SCHEDULER_SYNC` su sm_89.
Esecuzione delle fixture scheduler CUDA SKIP (exit 77): nessun device.
La build sm_121 su x86_64 verifica il codice GPU; il binario DGX Spark
richiede ancora host aarch64 e librerie corrispondenti.
**Aperto:** gate CUDA/Compute Sanitizer e benchmark separati Ada/GB10;
poi Graphs e specializzazione ADE con confronto misurato, oltre alla pipeline
prepare/postprocess nativa.

---

## 2026-10-09 — CUDA Graphs, stencil/ADE fusi e harness nativo
**Contesto:** lo scheduler a eventi elimina le attese host per passo ma lascia
i lanci individuali; rigid+FDGrid conserva una pressione intermedia globale.
**Modifiche:** `PFFDTD_GRAPHS=1` cattura pacchetti da 96/6 passi sul launcher
comune, con cache per fase n%6, contatore device sorgenti/readout e code dirette
prima dei drain a 512 campioni. Setup/cattura resta incluso nel wall time.
`PFFDTD_BOUNDARY_FUSED=1` unisce stencil e ADE con mappa boundary/lossy
validata int32/int64; tre carry e ordine aritmetico conservati. I riferimenti
sync, eventi e `BOUNDARY_SEPARATE` rimangono selezionabili.
Harness `fdtd_bench.cu` con HDF5 esistenti, confronti esatti, interleaving,
CSV/metadati e wall time completo di run_sim; nessun calcolo Python.
**Razionale:** grafi ripetibili eliminano lanci host individuali del pacchetto;
fusione completa evita store/load di pressione e indice lossy, al costo della
mappa per ogni boundary e della pressione sui registri. Nessuno speedup dedotto
dai soli conteggi o dalla compilazione.
**Esito:** PASS planner Graphs su 1440 modelli randomizzati e sanitizer;
PASS mappa su 4101 casi strict/sanitizer; PASS stencil/ADE FP32/64, 2820x37
per precisione anche ASan/UBSan. Fixture CUDA completa ampliata a 96 casi
per precisione; unit device boundary a 348x37, con stato/padding seeded.
PASS build CUDA/fixture Ada e GB10; esecuzione device SKIP77 senza GPU.
ptxas fuso generico: Ada 40/44 registri FP32/64; GB10 Cart/FCC FP32 48/64,
FP64 64; stack 96/192 B e zero spill dichiarati. Non è misura del traffico.
**Aperto:** bit equality/stabilità/RIR/Compute Sanitizer e benchmark Ada/GB10;
array ADE locali, pipeline prepare/postprocess nativa e temporal blocking.

---

## 2026-10-09 — ADE scalar reload e dispatch a poli fissi
**Contesto:** il report ptxas conferma array ADE locali: stack 96/192 B per
thread, pur senza spill dichiarati. Il solo conteggio spill non bastava.
**Modifiche:** con boundary fusa `PFFDTD_ADE_MODE=reload` usa due passaggi
scalari, senza aggiornare stati nel primo. `fixed` srotola i poli0/1/11/12
e usa reload negli altri casi; `generic` resta il default. Il benchmark
interleaved confronta dodici combinazioni con CSV, metadati e verifica esatta.
**Razionale:** reload rimuove gli array locali e rilegge gli stati globali;
fixed evita quelle riletture nei casi comuni ma aumenta la pressione sui registri.
**Esito:** PASS 56 casi ADE x37 per precisione contro CPU indipendente,
incluse rotazioni/padding/zero area/stati nulli con zero poli; PASS sanitizer.
I test stencil2820x37 e CUDA348x37 includono tutti e tre i modi e poli0..12.
ptxas integrato, map32/64: Ada reload39/44 registri FP32/64, fixed96/166;
GB10 reload48 FP32 e48Cart/56FCC FP64, fixed80/166. Stack/spill zero per
reload/fixed nelle48 istanze dei quattro build, rispetto a96/192 B generic.
Build CUDA/fixture/benchmark PASS; device SKIP77, nessuna misura di velocità.
**Aperto:** scegliere per GPU e densità solo dopo gate numerici e benchmark;
stabilità lunga/RIR, budget memoria, temporal blocking e pipeline nativa.

---

## 2026-10-09 — Generatore nativo di scene HDF5 per benchmark
**Contesto:** il harness CUDA richiedeva input già preparati; il vincolo vieta
calcolo Python anche nella preparazione delle nuove scene di prova.
**Modifiche:** CLI `fdtd_fixture.cpp` in C++/HDF5, target `fixture` e
`test-fixture` incluso in `test-native`. Pannelli Cartesian reciproci,
materiali RLC passivi con poli0..12, sorgenti distinte fuori boundary/ABC,
ricevitori e metadati completi. Parametri per dimensioni, Nt, frazione rigida,
materiali misti e densità dei pannelli. Verifica col loader originale e
rifiuto directory esistenti; nessun calcolo Python.
**Razionale:** rendere eseguibili i confronti Ada/GB10 con geometrie analitiche
controllate, variando il costo ADE senza dipendere dalla voxelizzazione mesh.
**Esito:** PASS26 round trip HDF5 per precisione anche ASan/UBSan.
Smoke solver CPU originale FP32/FP64: scena8x9x10,3073 passi,13 materiali,
Ns2/Nr6/Nb24/Nbl16,OMP2,exit0. Checker C++ conferma `u_out[6,3073]`:
18.438 campioni finiti e18.420 non nulli per ciascuna precisione.
**Aperto:** il generatore è analitico Cartesian, non una migrazione generale
di mesh/FCC/fitting/postprocess. Gate CUDA, stabilità/RIR e misure per target
restano aperti; nessun guadagno prestazionale dedotto dalla prova CPU.
