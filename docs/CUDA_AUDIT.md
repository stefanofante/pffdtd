# Analisi critica PFFDTD: CUDA su RTX 4500 Ada e DGX Spark GB10

Analisi del 9 ottobre 2026, commit `ad169d2e9e13b5072236fbe1b5bdda023a743501`, coincidente con `origin/develop` al momento dell'ispezione. Obiettivo: calcolo nativo C++/CUDA, inclusi preparazione e post-elaborazione; nessuna simulazione o elaborazione numerica Python eseguita.

## Valutazione

Il forward CUDA è una base sensata: accessi contigui nel bulk, FCC ripiegato, due campi della pressione, stato delle impedenze in struttura per polo, input caricati una volta, output a blocchi, indici bulk a 32 bit quando sicuri. Non conviene sostituire tutto contemporaneamente e perdere il riferimento numerico. Conviene riscrivere progressivamente scheduler, boundary e pipeline completa; valutare un nuovo discretizzatore come ramo separato con un contratto di accuratezza.

La priorità più urgente è una dipendenza lettura/scrittura nella recente fusione degli halo. Seguono eliminazione delle barriere host e riduzione dei passaggi boundary. I guadagni quantitativi richiedono GPU: nessuna accelerazione è stata misurata in questo ambiente.

## Evidenza disponibile e limiti

- Ispezionati README, Makefile, tutti gli header e il main C/CUDA, flusso Python di setup, voxelizzazione, fit materiali, solver e post-elaborazione, diario e benchmark disponibili.
- GCC/OpenMP e runtime HDF5 presenti; header HDF5 installati localmente da pacchetto Debian autenticato tramite APT. Compilazione CPU FP32 e FP64 riuscita; collegamento dinamico completo.
- Riproduttore C della dipendenza halo compilato ed eseguito, descritto sotto.
- GPU, driver esposti, `nvcc`, Nsight Systems e Nsight Compute assenti. Non eseguiti kernel CUDA, Compute Sanitizer, benchmark, confronti RIR o simulazioni CPU complete.
- I risultati del diario sono evidenze riportate dal repository, non nuove misurazioni riprodotte qui. Il CSV dei benchmark riguarda prevalentemente GPU precedenti ai due target.
- Non rilevata una pipeline CI tracciata con gate CUDA automatici. Gli script chiamati `test_script_*` sono soprattutto esempi di setup/simulazione e visualizzazione, non una suite autonoma di regressione CUDA.

## Problemi e opportunità, con priorità

### P0 — Fusione halo: scritture disgiunte ma dipendenza RAW

Fonte: [gpu_engine.h:535](../c_cuda/gpu_engine.h#L535), kernel `FlipHaloFaces`.

La faccia XZ esegue `u1[z*NxNy+x] = u1[z*NxNy+x+2*Nx]` per `1 <= x < Nx-1`. La faccia YZ, per `y=0`, legge `u1[z*NxNy+2]` per scrivere il corner `x=0,y=0`. Quel valore è proprio una destinazione di XZ, aggiornata dalla prima faccia. Analoga dipendenza esiste all'estremo `x=Nx-1`.

La versione separata ordina XZ prima di YZ. La versione fusa assegna le facce a blocchi diversi tramite `blockIdx.z`, senza ordine globale. Saltare i corner nelle scritture XZ elimina il conflitto write/write ma non il conflitto read/write. `__syncthreads()` non risolverebbe una dipendenza fra blocchi. Il ragionamento di correttezza B11 nel diario è quindi incompleto.

Riproduttore: modello C host delle tre facce. Su griglia 8x8x4 inizializzata con valori distinti, l'ordine separato produce corner 19, l'ordine YZ-prima-di-XZ ammesso dal kernel fuso produce 3. Un terzo caso legge direttamente il punto interno composto, e coincide con la sequenza separata sull'intero array. Sono prove host della dipendenza e della correzione per questa fase isolata; non sono un test eseguito su GPU.

Lo stencil Cartesiano assiale non legge i corner a due facce: una prova Cartesiana può quindi non rilevare il problema. FCC legge il corner dalla cella `(1,1,z)` tramite offset `-Nx-1` ([gpu_engine.h:282](../c_cuda/gpu_engine.h#L282)). Una simulazione breve con campo nullo alla periferia può nasconderlo anche in FCC.

Correzione proposta: baseline con `-DHALO_SEPARATE`; poi fusione con letture dei corner direttamente dalla composizione delle riflessioni interne, senza leggere destinazioni concorrenti. Validare anche le altre facce condizionali, FCC fold, dimensioni minime, spigoli e partizioni multi-GPU. Non dichiarare bit-identità sulla sola base delle scritture disgiunte. La correzione CUDA non è stata applicata durante questa analisi.

### P1 — Scheduler host seriale a ogni timestep

Fonti: [gpu_engine.h:1091](../c_cuda/gpu_engine.h#L1091), [gpu_engine.h:1225](../c_cuda/gpu_engine.h#L1225), [gpu_engine.h:1271](../c_cuda/gpu_engine.h#L1271), [gpu_engine.h:1296](../c_cuda/gpu_engine.h#L1296).

Un solo thread host lancia il lavoro di tutte le GPU. Ogni passo sincronizza entrambi gli stream, scambia gli halo, sincronizza ancora, interroga eventi e stampa progresso. Anche a una GPU rimangono sincronizzazioni dello stream boundary dopo una fase di scambio senza trasferimenti. Il readout a 512 passi riduce le copie D2H, ma non elimina queste barriere.

Riscrittura proposta: CUDA Graphs con dipendenze esplicite, stream non bloccanti, telemetria ogni blocco di passi e cronometro host monotono per il tempo complessivo. Separare grafi per Cart/FCC, precisione e topologia; evitare aggiornamenti host di tutti i nodi a ogni passo.

La rotazione bulk ha periodo 2, quella boundary periodo 3: un grafo a indirizzi fissi può ripetere un ciclo di 6 passi. Occorre trattare esplicitamente contatore dei campioni, posizione nel buffer output, blocchi finali incompleti e riuso degli eventi. Non basta catturare un passo e rilanciarlo con puntatori e `n` ormai obsoleti. Per multi-GPU usare grafi per dispositivo collegati con eventi/comunicazioni supportate e verificarne l'ordine.

Potenziale maggiore per griglie piccole o numerosi passi; per griglie grandi dominate dalla memoria il beneficio globale può essere limitato. Misurare gap GPU e tempo host prima di scegliere la dimensione del batch.

### P1 — Pipeline boundary con intermedi e gather/scatter

Fonte: [gpu_engine.h:1105](../c_cuda/gpu_engine.h#L1105), [gpu_engine.h:398](../c_cuda/gpu_engine.h#L398).

Sequenza corrente: rigid su tutti i nodi boundary → gather dei nodi dissipativi → correzione ADE → scatter alla griglia. Tre passaggi intorno al calcolo ADE movimentano pressione e indici; lo stencil bulk attende la fine dell'intera sequenza.

Proposta: separare boundary rigidi e dissipativi, eseguire per i dissipativi stencil + correzione ADE + aggiornamento degli stati + scrittura finale in un kernel, conservando in registri gli intermedi. Precalcolare i coefficienti geometrici immutabili e valutare specializzazione per numero di poli. Conservare il valore precedente necessario alla ricorrenza prima della sovrascrittura, con timeline esplicita degli stati. La rimozione di `u1b` da sola è già stata correttamente scartata dal diario perché aggiungerebbe un gather.

Questa fusione è diversa dall'eliminazione del padding: il padding non è il traffico dominante. La fusione elimina passaggi reali; non elimina automaticamente la lettura/scrittura degli 11 poli per cella.

### P1 — Stato ADE: verificare spill e specializzare con criterio

`vh1int[MMb]` e `gh1int[MMb]` sono array locali indicizzati da loop a limite dinamico `cuMb[k]`. Possono finire in registri oppure memoria locale a seconda del compilatore; non è lecito affermare che ci siano spill senza PTX/SASS e contatori. Richiedere `ptxas -v`, Nsight Compute e ispezione di local loads/stores.

Kernel specializzati `M=11`, loop srotolati e coefficienti omogenei per blocco possono migliorare scalarizzazione e accessi ai coefficienti. Raggruppare solo per materiale può però peggiorare la località spaziale dei gather della pressione: preferire tile spaziale + sottogruppo materiale e confrontare entrambi. La constant memory è conveniente con accessi uniformi; materiali diversi nella stessa warp possono serializzare il broadcast. Non spostare indiscriminatamente tutti i coefficienti in constant memory.

Per tagliare davvero il traffico degli stati occorre ridurre l'ordine dei modelli: fit passivi a meno poli con errore controllato sulla banda e sugli angoli d'incidenza, poi gate su decadimento e RIR. Questo cambia la rappresentazione fisica e richiede validazione distinta dalla bit-identità.

### P1 — Gestione memoria descritta ma non implementata

README righe 94–96 dichiara `cudaMemGetInfo` e margine adattivo. Nel codice CUDA esistono `totalGlobalMem` e una variabile `totalmembytes` popolata ma non usata per budget; non compare una chiamata `cudaMemGetInfo`. Le allocazioni sono `cudaMalloc`, non `cudaMallocManaged`.

Proposta: calcolo del fabbisogno prima delle allocazioni, query della memoria effettivamente libera, margine documentato, overflow checking e diagnosi utile. Distinguere memoria fisica condivisa GB10 da managed memory CUDA: non assumere paging illimitato o degradazione dolce solo perché il SoC condivide RAM. Lasciare memoria a host, driver e preprocessing; controllare il limite reale del container.

Fabbisogno principale per GPU, con `s=sizeof(Real)`, `B=512`, `N` celle locali inclusi halo:

`2*N*s + ceil(N/8) + Nb*(8+2+1) + Nbl*(8+1+4*s+2*MMb*s) + Nba*(8+1+s) + Nr*(8+B*s) + Ns*(8+8*Nt) + Nm*s + Nm*MMb*sizeof(MatQuad)`.

È una stima delle allocazioni esplicite device; aggiungere contesti, risorse dei grafi, buffer aggiuntivi e margine. Host: tutti gli input, `Nr*Nt*8` output, buffer pinned e temporanei; il loader costruisce inoltre `bn_mask_raw` da un byte per cella per verificarne la versione compressa.

### P1 — Multi-GPU: partizione e comunicazione

Fonte: [gpu_engine.h:595](../c_cuda/gpu_engine.h#L595), [gpu_engine.h:1239](../c_cuda/gpu_engine.h#L1239).

Partizione uniforme in numero di slab, senza pesare densità dei boundary, poli ADE, capacità/banda dei dispositivi o topologia. Tutte le GPU visibili vengono usate: più GPU possono peggiorare il tempo se il sottodominio è piccolo o la comunicazione domina. P2P viene tentato fra tutte le coppie, pur comunicando solo con i vicini; non esiste una pipeline esplicita di staging pinned controllata dall'applicazione quando P2P è assente.

Proposta: costo per slab stimato da bulk + boundary/poli + comunicazione, assegnazione vincolata alla memoria libera, scelta del numero di GPU e vicinanza topologica. Separare interior e skin; produrre subito le skin da trasferire, sovrapporre comunicazione e bulk, attendere solo dove lo stencil seguente richiede gli halo. Conservare un percorso a una GPU per entrambi i target. Non aggiungere multi-GPU al GB10 come prerequisito di prestazione.

### P2 — Output: batching corretto come idea, ancora bloccante

Fonte: [gpu_engine.h:1208](../c_cuda/gpu_engine.h#L1208).

Al drain si copia su host, si sincronizza e si trasforma il layout in doppi sulla CPU. Proposta: buffer device/pinned doppi con stream I/O dedicato, conversione e ricombinazione dei ricevitori su GPU, scrittura HDF5 per chunk senza mantenere sempre `Nr*Nt` doppi su host. Le otto componenti di interpolazione di un ricevitore vanno ricombinate preservando associazioni e ordine di riduzione. Evitare FP64 come formato obbligatorio di trasferimento per simulazioni FP32, salvo requisito di output.

### P2 — Robustezza e strumenti di misura

- `cudaGetDeviceCount` non viene controllato e non c'è guard `ngpus>0` prima della divisione in `split_data` ([gpu_engine.h:742](../c_cuda/gpu_engine.h#L742)). Un errore di driver/assenza GPU deve produrre una diagnosi, non dividere per zero.
- `check_sorted` viene eseguito solo con più GPU, mentre `AddInBatch` assume indici sorgente distinti. Se nuovi input nativi consentono duplicati, definire consolidamento deterministico o riduzione; evitare una race da `+=` concorrenti.
- `scale_input` divide per `max_in` senza gestire input tutto nullo ([fdtd_data.h:879](../c_cuda/fdtd_data.h#L879)). Gestire il caso nullo e dati non finiti prima del lancio.
- `print_progress` usa `winsize` non inizializzato se `ioctl` fallisce e calcola `Nt/n` anche per `n=0` ([fdtd_common.h:110](../c_cuda/fdtd_common.h#L110)). In log senza TTY può influire sull'affidabilità del runner. Va separato dal percorso prestazionale.
- I tempi air includono gather ABC, halo, sorgenti e ABC: non sono tempi puri dello stencil. I tempi boundary e i breakdown vengono letti solo per GPU0: non descrivono lo sbilanciamento fra GPU.
- Errori CUDA/HDF5 e dimensioni vengono spesso gestiti tramite assert. Non compilare con `NDEBUG` come ottimizzazione finché assert essenziali non vengono sostituiti da validazione esplicita. Aggiungere controlli overflow a conteggi e byte.
- `-arch=native` non produce un artefatto universale Ada/GB10 e richiede hardware visibile. Distribuire build separate o fatbin con architetture esplicite compatibili con la versione del toolkit. Le librerie HDF5 sono specifiche anche per x86_64/aarch64.

## Ripensare gli algoritmi che cambiano il costo dominante

### Bulk FDTD e temporal blocking

Lo stencil oggi legge il campo corrente, legge il campo precedente e scrive il futuro. Con riuso ideale il limite inferiore bulk è circa `3*N*s` byte per passo, più maschera e overhead: 12 byte/cella FP32 o 24 FP64. I 6/12 accessi vicini non equivalgono automaticamente a 6/12 letture DRAM, grazie alle cache.

Il diario stima circa 220 GB/s sul GB10 per il bulk grande; è un lower-bound analitico, non un contatore DRAM. Se fosse già vicino al limite sostenibile, semplici modifiche del blocco o un tile spaziale non possono fornire multipli di accelerazione. Il tiling spaziale B9 è documentato come più lento sul GB10: non ripeterlo senza una nuova ipotesi.

La leva strutturale è riutilizzare dati per più passi temporali: tile temporali con dominio di dipendenza allargato, lavoro ridondante sugli halo e scrittura dei soli punti validi. Evitare aggiornamenti in place che leggano valori del timestep futuro da blocchi vicini; usare fasi globalmente ordinate oppure domini temporali con input immutabili e output separati. Shared memory o TMA possono aiutare il movimento dati, ma non sostituiscono questa dimostrazione di dipendenza.

Partire da tile interni senza boundary, sorgenti o ricevitori, poi introdurre skin e nodi speciali. La frontiera si allarga a ogni passo; costi di halo, occupazione, registri e ADE possono annullare il risparmio. Kernel persistenti con grid sync richiedono una griglia cooperativa con vincoli di residenza; non sono una soluzione universale per scene da centinaia di milioni di celle.

### Meno celle e meno timestep a parità di errore fisico

Per dominio e durata fissi, ridurre il passo `h` aumenta le celle circa come `h^-3` e i timestep circa come `h^-1`: costo volumetrico circa `h^-4`, salvo geometria e vincoli. Ridurre i punti per lunghezza d'onda mantenendo la dispersione entro tolleranza può dare più beneficio di un micro-kernel, ma cambia la discretizzazione.

Confrontare Cart/FCC e stencil di ordine superiore sulla stessa tolleranza di fase, non sullo stesso numero di celle. Gli stencil più larghi aumentano traffico, halo e complessità dei boundary; il CFL può compensare o annullare il vantaggio. Una riscrittura high-order richiede nuovo trattamento coerente di impedenze, correzione geometrica e stabilità, non solo coefficienti diversi nel bulk.

Mesh adattive o solver DG possono ridurre vuoto/oversampling e migliorare la geometria, ma cambiano il progetto numerico, la regolarità GPU e spesso il passo temporale globale. Sono candidati per un prototipo separato, non accelerazioni garantite di questo solver.

### Spazio attivo e scene sparse

Il bulk percorre il bounding box. Per scene con molti volumi inutili, classificare tile attivi, interni e boundary, usando indici locali contigui. Non scartare semplicemente celle "esterne": il modello permette superfici non watertight, aperture e propagazione oltre alcuni oggetti. La maschera attiva deve rispettare la fisica della scena e i boundary assorbenti. Scene dense possono peggiorare con indirezione e compattazione: serve una soglia di selezione misurata.

### Pipeline completa senza calcolo Python

Il solver CUDA è solo una parte del flusso: il repository calcola geometria, voxelizzazione, fit dei materiali, ricombinazione, filtri e resampling in Python/NumPy/Numba/SciPy. Il vincolo delil progetto richiede migrazione anche di questi componenti.

Architettura proposta: CLI C++ per configurazione e HDF5/JSON; preparazione CUDA; forward CUDA; postprocess CUDA. C++ host gestisce file, controlli, orchestrazione e metadati. Le verifiche numeriche possono essere C++/CUDA e analytic fixtures, senza dipendenza Python. Il solver CPU C esistente può servire come riferimento indipendente quando la semantica di arrotondamento lo consente.

Voxelizzazione: BVH dei triangoli, query segmento-triangolo nei 6/12 link dello stencil, scelta del hit/materiale, correzione delle aree e verifica della reciprocità delle adiacenze. Valutare OptiX per sfruttare RT cores rispetto a BVH CUDA; introdurrebbe una dipendenza da SDK/versioni. Una query di triangoli più rapida non basta: preservare tolleranze, sidedness, link vicini alle superfici, orientamento e regole per scene aperte. Organizzare tile e stream in batch per non materializzare tutti gli array intermedi della versione Python.

Materiali: riutilizzare inizialmente gli HDF5 già presenti; portare successivamente il fitting passivo a C++/CUDA, con ricerca a ordine ridotto e gate fisico. Postprocess: riduzioni pesate dei ricevitori, filtri SOS e resampling polifase; filtri aria in batch/FFT dove appropriato con cuFFT. Le ricorrenze IIR sono sequenziali nel tempo: parallelizzare su ricevitori/sezioni o derivare una scan affine validata, non usare una convoluzione diversa senza controllarne l'equivalenza.

Una CLI nativa con subcomandi prepare/run/postprocess/validate/benchmark rende riproducibile il flusso; cuBLAS/cuFFT/RT cores vanno usati nei componenti che corrispondono ai loro algoritmi. Il bulk stencil non diventa automaticamente più rapido riscrivendolo come GEMM su Tensor Cores.

## Politica per i due hardware

| Area | RTX 4500 Ada | DGX Spark GB10 |
|---|---|---|
| Esecuzione | Memoria device dedicata, budget rigoroso per GPU | RAM fisica condivisa; budget con host e limiti sistema |
| Build | Binari/toolchain per macchina Ada | Build aarch64 e toolkit compatibile Blackwell GB10 |
| Bulk | Misurare working set/cache e temporal blocking | Default bulk già competitivo secondo diario; evitare slab B9 senza nuova evidenza |
| Boundary | Misurare spill/occupazione e fusione ADE | Stesse prove; bilanciare traffico con bulk su memoria condivisa |
| I/O | PCIe e pinned double buffering | Evitare copie ridondanti, ma misurare accessi e contesa host/device |
| Selezione | Varianti misurate per dimensione e sparsità | Varianti autonome; nessun vincitore imposto dall'altro target |

Non occorrono due implementazioni fisiche completamente divergenti: numerica comune, politiche di lancio/memoria e specializzazioni per target. Non hardcodificare VRAM, architettura o soglie sulla base del solo nome commerciale.

## Gate e piano operativo

1. **Baseline affidabile:** ripristinare semanticamente la fase halo ordinata o correggere le letture composte; aggiungere fixture native con valori non nulli a facce/spigoli. CUDA FP32/FP64, Cart/FCC, una/più GPU, casi minimi e output con blocchi parziali. Build e Sanitizer in hardware compatibile; racecheck da solo non garantisce di rilevare race su memoria globale.
2. **Misura riproducibile:** harness C++ con scene piccole/grandi/sparse, densità boundary variabile e diversi numeri di poli, warmup, ripetizioni interleaved, mediana e variabilità. Registrare GPU, toolchain, clock/power/temperatura, dimensioni, precisione e commit. Nsight Systems per idle/sync/I/O; Nsight Compute per DRAM/L2/register/local memory. Se i contatori sono vietati, riportarlo senza trasformare stime in misure.
3. **Scheduler:** Graphs e telemetria a batch; confronto end-to-end, conservando ordini e campioni esatti. A una GPU prima, multi-GPU dopo.
4. **Boundary:** fusione e specializzazioni con gate numerico separato; confrontare byte e kernel-time oltre al throughput finale.
5. **Pipeline nativa:** prepare/postprocess C++/CUDA, mantenendo lo schema HDF5 per interoperabilità. Validare reciprocità della mesh, passività, filtri e layout ricevitori con fixture native.
6. **Algoritmi:** temporal blocking e riduzione dell'ordine ADE; poi stencil high-order/adattività come esperimenti separati con stesso budget d'errore e confronto sul tempo per secondo fisico simulato.

Per modifiche di scheduling/layout che preservano ordine aritmetico usare confronto esatto quando applicabile. CPU e GPU non sono necessariamente bit-identici a causa degli intrinsics di rounding e FMA. Per nuovi algoritmi fissare tolleranze motivate su fase/dispersione, ampiezza, energia, passività, stabilità a lungo termine, RIR/decadimento e ricevitori; non accettare soltanto un errore medio che nasconda instabilità tardive. FP16, fast-math, reassociazione e meno poli vanno trattati come cambiamenti numerici, non come flag innocui.

## Validazione ancora necessaria

Per completare i gate CUDA occorrono dispositivi NVIDIA e toolchain compatibili su entrambi i target. Il successo della compilazione host e del modello di dipendenza non valida l'esecuzione GPU, le RIR o i guadagni prestazionali. Il flusso numerico nativo completo richiede ancora la migrazione di prepare e postprocess.
