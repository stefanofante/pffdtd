# Analisi critica PFFDTD: CUDA su RTX 4500 Ada e DGX Spark GB10

Analisi del 9 ottobre 2026, commit `ad169d2e9e13b5072236fbe1b5bdda023a743501`, coincidente con `origin/develop` al momento dell'ispezione. Obiettivo: calcolo nativo C++/CUDA, inclusi preparazione e post-elaborazione; nessuna simulazione o elaborazione numerica Python eseguita.

Stato dopo gli interventi: il percorso JSON mesh/materiali DEF → preparazione
Cart/FCC → forward → DSP/resampling/aria → HDF5/WAV è disponibile in C++/CUDA.
La preparazione usa BVH residente e compattazione stabile; il filtro modale ha
una variante FFT con errore d'interpolazione controllato. Sono passati test
nativi e prove su mesh reali. Le build sm_89/sm_121 sono verificabili su questo
host; esecuzione CUDA, speedup e validazione acustica sulle due GPU rimangono
da misurare. I finding iniziali sotto sono conservati come contesto storico;
gli aggiornamenti successivi specificano cosa è stato corretto.

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

## Interventi successivi all'audit

La correzione halo legge direttamente i punti con entrambe le riflessioni applicate agli spigoli `y=0`. La funzione condivisa `c_cuda/halo_faces.h` permette di verificare che tutte le destinazioni siano uniche e che nessuna sorgente appartenga al write-set. Per assi di tre celle il runtime conserva le facce ordinate. `tests/native/halo_faces_test.cpp` copre 640 casi host, FP32/FP64, int32/int64, Cart/FCC, fold, estremi delle slab, ordini dei volti e interleaving dei thread, oltre agli indici int64 grandi. Compilando lo stesso file con nvcc si esercita anche la funzione su device.

La pipeline dissipativa ora offre un kernel unico che legge la griglia, applica la stessa ricorrenza ADE e scrive pressione finale e carry. Il rigid stencil, i tre carry e l'ordine dei calcoli sono conservati. `BOUNDARY_SEPARATE` mantiene l'originale gather/ADE/scatter come riferimento indipendente. Si eliminano due lanci e, dal conteggio degli accessi espliciti, `3*sizeof(Real)+sizeof(int64_t)` byte per nodo e timestep: 20 byte in FP32, 32 in FP64. È una riduzione degli accessi richiesti, non una misura del traffico DRAM o dell'accelerazione.

`tests/boundary_fd_check.c` confronta il percorso condiviso con il solver ADE CPU esistente su dieci casi e 37 passi per precisione, inclusi tutti i bank della pressione, gli stati usati e il padding. `tests/boundary_fd_cuda_check.cu` confronta i due kernel CUDA con 257 nodi, più blocchi e gli stessi periodi di rotazione, senza fixture Python.

Il Makefile consente architetture esplicite e output fuori checkout, oltre a test nativi e CUDA. I test CUDA segnalano assenza hardware con exit 77; la loro compilazione non equivale all'esecuzione. Le build `sm_121` effettuate su x86_64 validano il codice device GB10; per l'eseguibile DGX Spark occorre ancora la build host aarch64. Il runtime controlla ora l'esito dell'enumerazione CUDA prima di partizionare la griglia. Il README è allineato all'attuale gestione memoria, ancora priva di budget adattivo.

I gate hardware, i benchmark e il nuovo scheduler restano aperti. Per CUDA Graphs va gestita anche la fase del readout a 512 campioni: il solo periodo di sei passi dei puntatori non chiude tutte le fasi (LCM 1536), salvo contatori/nodi aggiornati o varianti del grafo progettate esplicitamente.

### Scheduler asincrono a una GPU

Il percorso `PFFDTD_ASYNC=1` è ora implementato come catena di eventi: boundary(n) attende air(n-1), air(n) attende boundary(n), readout(n) rimane nello stream boundary prima del lavoro del passo successivo. La rotazione host dei puntatori non modifica gli argomenti dei lanci già accodati. Stream non bloccanti e una sincronizzazione iniziale rendono esplicita la dipendenza dalle inizializzazioni. Il drain ogni 512 passi o a fine simulazione attende entrambi gli stream prima di copiare e riusare il buffer.

Si eliminano le tre sincronizzazioni stream e l'attesa dell'evento di timing che il percorso precedente effettua a ogni passo. Rimangono i lanci dei kernel e gli eventi di dipendenza: questa modifica non è ancora CUDA Graphs. La telemetria asincrona riporta il wall time a blocchi e non inventa un breakdown air/boundary. `PFFDTD_PROGRESS=0` permette prove senza output di progresso. Il percorso multi-GPU rimane sincrono; ricevitori non interni fanno ripiegare sul riferimento. `SCHEDULER_SYNC` mantiene il riferimento anche quando l'opzione runtime è impostata.

`tests/native/scheduler_plan_test.cpp` verifica 576 ordini topologici randomizzati con bank bulk/carry e campioni etichettati; rileva mutanti senza le dipendenze necessarie, copre indici temporali int64 e drain parziali. È una verifica del contratto delle dipendenze, non dell'API eventi CUDA. `tests/scheduler_cuda_check.cu` usa il solver completo con fixture C++ native, due sorgenti, ricevitori duplicati/interni, ABC, pannello con adiacenze reciproche e stato ADE. Confronta esattamente i campioni sync/async per Cart/FCC, FP32/FP64 e Nt=1,2,3,5,6,7,511,512,513,1025. Senza hardware esce con 77; la regressione GPU non è stata dichiarata passata.

### CUDA Graphs riutilizzabili

`PFFDTD_GRAPHS=1` usa lo stesso launcher della fisica dei percorsi sync/eventi, catturato su un unico stream non bloccante. I grafi da 96 e 6 passi chiudono le rotazioni bulk2/carry3; la cache è distinta per lunghezza e fase `n%6`. Le code dirette cambiano la fase dopo il drain a 512 campioni, quindi il solo grafo della fase iniziale non sarebbe sufficiente. Sorgenti e readout leggono un contatore device più offset; il contatore avanza dopo tutti i nodi del pacchetto. Il planner impedisce di attraversare il drain o il limite Nt. Nel blocco pieno si accodano cinque grafi da 96, cinque da 6 e due passi diretti. Il numero di API host si riduce strutturalmente; non è una misura dello speedup.

Il readout nel grafo segue air e legge il campo `u1` interno, immutato da air; il controllo ricevitori protegge questa equivalenza. Un solo stream sacrifica la sovrapposizione readout/air del percorso a eventi e semplifica le dipendenze; confrontare i due sul device. La sincronizzazione iniziale include il contatore e tutto il setup sul default stream. Il costo di cattura/istanziazione è riportato separatamente come durata host e rimane incluso nel wall time, anche se parte può sovrapporsi a lavoro GPU già accodato. Cleanup dopo il completamento dei grafi, prima della liberazione dei buffer.

`graph_plan.h` e il test nativo verificano 1.440 modelli randomizzati host/device: fasi dei puntatori, versioni sorgente/readout, riuso dell'output, code int64 e mutanti con contatore/fase errati. PASS anche ASan/UBSan. Il test completo CUDA è ampliato a 96 fixture per precisione, incluse Nt=95/96/97 e 1536/1537/3073, ricevitori direttamente sulle boundary, materiali con 0..12 poli e boundary rigide/dissipative miste. Ogni fixture confronta undici varianti col riferimento sincrono: 1.056 confronti per precisione. Le comparazioni device rimangono da eseguire.

### Fusione completa dello stencil boundary e ADE

`PFFDTD_BOUNDARY_FUSED=1` evita la pressione intermedia globale fra rigid stencil e ADE. Una mappa validata associa ogni boundary al suo ordinal dissipativo, o -1 per rigido; preserva gli ordini originali, usa int32 quando basta e int64 altrimenti. Indici duplicati, fuori griglia o dissipativi assenti dalla boundary sono errori espliciti. Gli input ordinati usano una costruzione lineare; solo quelli disordinati richiedono copie ordinate. Si conserva il layout degli stati, l'aritmetica e la rotazione dei tre carry. `BOUNDARY_SEPARATE` forza il riferimento.

Rispetto a rigid + FDGrid si elimina un lancio e si risparmiano nominalmente `(8+2*sizeof(Real))*Nbl - sizeof(MapIdx)*Nb` byte per passo: si rimuovono indice dissipativo e store/load della pressione, aggiungendo la mappa per ogni boundary. Per poche boundary dissipative il bilancio può diventare negativo. La memoria device cresce di `Nb*sizeof(MapIdx)`; il setup host aggiunge il buffer e lo staging della mappa, oltre alle coppie indice/ordinale per input disordinati. Si tratta di accessi espliciti e memoria allocata, non di traffico DRAM misurato.

PASS 4.101 casi della mappa e, per precisione, 2.820 casi stencil/ADE x37 passi con riferimento stencil indipendente e ADE CPU esistente; PASS ASan/UBSan. Il test CUDA dedicato prepara 257 boundary, campi e stati non nulli e confronta interamente griglia, carry, poli inattivi e padding: 348 casi x37 passi per precisione, Cart/FCC, map32/64, tre modalità ADE e tutti i poli 0..12. Compilazioni Ada/GB10 riuscite; esecuzioni SKIP77 senza device.

Il report ptxas CUDA13 per il kernel fuso generico mostra 40 registri FP32 e 44 FP64 su Ada; su GB10 48 Cart FP32, 64 FCC FP32 e 64 FP64. FDGrid usa rispettivamente 40/40 e 48/64. Gli array ADE mantengono 96 B di stack FP32 e 192 B FP64; zero spill dichiarati non equivale all'assenza di memoria locale. Il maggiore costo in registri FCC sul GB10 rende obbligatorio un confronto di occupazione e tempo, anche se si elimina un lancio.

### Harness nativo per scegliere le varianti

`fdtd_bench.cu` legge input HDF5 già preparati, confronta varianti interleaved e rifiuta campioni non finiti o diversi dal riferimento sincrono. L'output viene prima inizializzato a NaN per rilevare scritture mancanti. Riporta mediana, minimo, massimo e dispersione, con GPU, CUDA, precisione, griglia, conteggi e revisione di build in CSV. La metrica comune è il wall time esterno di `run_sim`, inclusi setup, trasferimenti, grafi, cleanup e reset; HDF5, scaling e verifica sono esclusi. I tempi del loop hanno clock diversi e sono etichettati separatamente. Ogni run ricrea il contesto: il warmup non conserva allocazioni o grafi. Clock, potenza e temperatura devono essere raccolti esternamente e non sono misurati dal harness.

Il confronto richiede una GPU visibile, ricevitori interni e sorgenti distinte. Non scrive gli HDF5 di output; il CSV richiede una scelta esplicita per sovrascrivere e protegge gli input. Assenza di device restituisce 77. Questo rende il benchmark eseguibile su entrambi i target senza calcolo Python, ma non sostituisce i gate numerici e di stabilità.

### ADE senza array locali: reload e poli fissi

Con la fusione boundary attiva, `PFFDTD_ADE_MODE=reload` conserva solo due scalari per polo: il primo passaggio calcola la pressione senza modificare stati, il secondo rilegge gli stessi stati globali e li aggiorna. Ogni nodo possiede ordinal e slot distinti; non legge valori aggiornati da altri nodi. Si elimina lo stack degli array dinamici al costo di due riletture globali per polo, potenzialmente servite dalla cache. `fixed` srotola 0/1/11/12 poli e usa reload per gli altri conteggi, mantenendo l'ordine della riduzione. Il default `generic` resta disponibile.

Il report ptxas per i kernel stencil/ADE integrati, con entrambe le larghezze della mappa, è il seguente. Reload e fixed hanno stack zero e nessuno spill dichiarato in tutte le build; generic ha 96/192 B di stack.

| Target | Precisione | Generic, registri | Reload, registri | Fixed, registri |
|---|---|---:|---:|---:|
| Ada sm_89 | FP32 | 40 | 39 | 96 |
| Ada sm_89 | FP64 | 44 | 44 | 166 |
| GB10 sm_121 | FP32 | 48 Cart / 64 FCC | 48 | 80 |
| GB10 sm_121 | FP64 | 64 | 48 Cart / 56 FCC | 166 |

Si tratta di conteggi del compilatore CUDA13, non di latenza, occupazione osservata o DRAM misurata. Fixed impone un costo elevato in registri anche ai nodi rigidi dello stesso kernel: non promuoverlo come scelta massima senza il benchmark. Il harness confronta dodici combinazioni scheduler/boundary/ADE con ordine Williams bilanciato su dodici round; il default è un warmup e dodici ripetizioni. Gli input nulli/non finiti e le sorgenti duplicate sono rifiutati.

`boundary_fd_specialized_test.cpp` confronta reload/fixed con ADE CPU indipendente su 56 casi x37 passi per precisione, inclusi tutti i poli, area nulla, materiali misti e zero poli con stati nulli: PASS anche ASan/UBSan. I test stencil completi e CUDA esercitano inoltre la fusione in tutti e tre i modi. I guadagni prestazionali e l'equivalenza sul device restano da misurare su ciascun target.

### Preparazione analitica HDF5 senza Python

`fdtd_fixture.cpp` genera in C++ i quattro input HDF5 del solver e del harness. La geometria è Cartesian con pannelli finiti: entrambi gli estremi dei link tagliati sono boundary, l'area segue i link rimossi e le adiacenze sono reciproche. I materiali RLC hanno coefficienti positivi e sono verificati dopo la discretizzazione tramite il loader originale. Sorgenti distinte fuori dai nodi boundary/ABC, segnali differenziati non nulli, ricevitori ordinati anche duplicati e metadati di interpolazione completano lo schema. Dimensioni, Nt, poli0..12, materiali misti, proporzione rigida e distanza dei pannelli sono parametri CLI. Directory esistenti sono rifiutate.

Questo permette scene riproducibili e densità boundary diverse senza eseguire i preparatori Python. Non sostituisce voxelizzazione di mesh, FCC, fitting di materiali misurati o postprocessing. Il generatore valida il round trip HDF5 e la posizione delle sorgenti anche con pannelli densi; non esegue il solver. Il target `test-native` comprende 26 casi per precisione, incluse code temporali e rifiuto overwrite: PASS FP32/FP64 e ASan/UBSan.

Smoke di integrazione nativo separato: scena8x9x10, Nt3073, tredici materiali con poli0..12, Ns2/Nr6/Nb24/Nbl16, solver CPU originale con due thread. PASS entrambe le precisioni: HDF5 `u_out[6,3073]`, 18.438 campioni finiti e 18.420 non nulli ciascuna. Il checker è C++/HDF5. Questa prova copre preparazione/loader/loop/scrittura output, senza certificare stabilità generale, RIR o CUDA. I benchmark sul device restano necessari.

### Preflight memoria e indici/lanci CUDA

Il solver ora confronta tutti i buffer espliciti con `cudaMemGetInfo` prima di allocare campi, mappa boundary e pinned output. Il budget comprende carry, padding ADE, sorgenti, readout, materiali, mappa opzionale e counter Graphs. Il pinned host è dichiarato a parte; overhead del runtime/grafi e resto della RAM host non sono inclusi. Il controllo non prenota memoria e non realizza una partizione adattiva. Sul GB10 occorre conservare anche il budget della RAM condivisa e del container.

Prodotti degli indici CUDA 1D e halo allargati prima della moltiplicazione evitano il wrap unsigned32 prima dell'assegnazione int64. Il percorso bulk/fold B7 mantiene Idx32 quando Npts lo consente. Le grid sono calcolate con ceil controllato e validate contro maxGridSize, maxThreadsDim e maxThreadsPerBlock prima della conversione a dim3 e delle allocazioni. Una griglia sottile Nx131073/Ny3/Nz3 può richiedere grid.z65536, oltre il limite65535, pur avendo pochi MB di campi. Ora riceve una diagnosi esplicita. PASS125 controlli nativi e ASan/UBSan; review indipendente del budget e delle forme di lancio. Nessun cambiamento all'aritmetica floating point.

### Postprocessing nativo, con parallelizzazione della ricorrenza

`fdtd_post.cpp` copre il percorso di default della CLI: ricostruzione pesata dei ricevitori → HP/integratore → resampling → LP opzionale. `u_out` scritto dal solver è già scalato e riordinato: applicare di nuovo out_reorder sarebbe un errore. `out_alpha[Nmic,K]` consolida Nr=Nmic*K righe; i due dataset finali `r_out_f` e `Fs_f` rimangono compatibili. Pubblicazione atomica in un nuovo HDF5, rifiuto overwrite implicito, protezione degli input/alias/symlink e metadati di filtro/backend/revisione.

Il design Butterworth supporta ordini1..16. Con diff1 il combinato parte da HP analogico2πfc e rimuove uno zero all'origine prima della bilineare; HP ordinario/LP usano prewarp. Cutoff nullo con diff1 usa l'integratore trapezoidale. LP simmetrico dimezza l'ordine ed esegue due passate a stato zero, senza padding filtfilt. Il design delle SOS può cambiare il rounding rispetto a SciPy. Il nuovo FIR sinc Kaiser ha coefficienti analitici condivisi CPU/CUDA, antialias per downsampling, zero extension e conteggi floor matematici esatti: non replica la tabella resampy bit per bit. Il costo per tap è elevato; LUT/polyphase sono candidati successivi da validare.

CUDA conserva dati e scratch in uno stream non bloccante fino al drain finale. IIR seriale è il riferimento. L'opzione chunked divide in64 campioni: prima calcola la risposta a stato iniziale zero, poi propaga i carry `s_next=T^64*s+B`, infine rifiltra ogni chunk col carry corretto. I chunk possono lavorare in parallelo su campioni disgiunti, anche in ordine reverse. Il tail usa lo stesso carry dai chunk completi; il suo stato finale non alimenta altri chunk. Due passaggi e tre kernel per SOS aggiungono lavoro/launch per esporre parallelismo temporale: la scelta richiede misure per Nt/Nmic e target, non un'affermazione di speedup.

PASS576 design contro TF analogica/bilineare indipendente,108 ricorrenze fino65537 campioni contro DFI long-double,20896 controlli FIR e fuzz indipendente2M casi. PASS modello affine/pipeline30.191.285 check fino100003 campioni/ordini1..16: errore relativo massimo al picco4.73e-10 contro seriale. PASS ASan/UBSan. CLI/HDF malformed, shape, metadata, alias e overwrite provati; scena nativa del solver3073 passi processata CPU con defaultHP10Hz8: PASS. Driver CUDA230 confronti, pesi diversi per canale, bypass/integrazione/HP/LP/reverse/rates/code e NaN-prefill, compilato Ada/GB10; esecuzione SKIP77. Il driver usa limite1e-12+1e-8*peak; `--verify` della CLI controlla ogni campione con limite più stretto1e-12+1e-8*abs(reference). Nessun confronto Python eseguito; gate device, RIR e velocità restano aperti.

### Altri finding fisici nella preparazione esistente

- [sim_consts.py:26](../python/fdtd/sim_consts.py#L26) usa `343.2*sqrt(Tc/20)`: a0°C produce velocità zero e sotto0°C un radicando negativo, pur ammettendo Tc≥−20. La formula Kelvin corretta, già in get_air_absorption, è `343.2*sqrt((Tc+273.15)/293.15)`. Il port nativo della preparazione dovrà usarla; la fixture analitica a temperatura fissa non è una migrazione di questo modulo.
- [vox_scene.py:632](../python/voxelizer/vox_scene.py#L632) verifica reciprocità con `assert ~(bitA ^ bitB)` su interi: ~0=−1 e ~1=−2 sono entrambi veri. La verifica non rileva mismatch. Il ramo FCC del link3 usa inoltre il vicino positivo anziché negativo e non applica correttamente il passo della parità. Il port deve interrogare link canonici e aggiornare entrambi gli estremi, verificando uguaglianza dei bit opposti.
- L'assorbenza Sabine non determina univocamente fase e impedenza. La riduzione dei poli richiede obiettivo fisico e passività/convergenza; importare e verificare i DEF esistenti permette di portare prepare prima di riscrivere il fitter.

Questi finding provengono dall'ispezione del codice, senza eseguire calcoli
Python. Sono affrontati nel port nativo descritto di seguito. Stokes è un
kernel gaussiano dipendente dal tempo; OLA e modal sono operatori distinti.
Una semplice convoluzione FFT non è una sostituzione equivalente.

### Preparazione di mesh reali in C++/CUDA

`fdtd_prepare.cpp` importa JSONRoomExport in metri, triangoli, sides, materiali,
sorgenti e ricevitori. Importa e valida i DEF HDF5 passivi esistenti, con al
massimo12 rami; non deduce una nuova impedenza complessa dall'assorbenza Sabine.
La velocità del suono usa Kelvin e conserva il margine CFL0.999; FP32 richiede
la sorgente differenziata con la vera ricorrenza bilineare. I corner di
interpolazione devono restare fuori boundary/ABC, anche quando il peso è zero.
Il preflight dei corner precede la voxelizzazione dell'intero volume: griglie
inadeguate ricevono una diagnosi col ricevitore, senza spostarlo automaticamente.

`mesh_geometry.h` precomputa FP64 e costruisce un BVH piatto con escape link,
foglie4 e ordinali deterministici. Una traversal per nodo valuta6/12 link,
riutilizzando il triangolo. I bounds conservativi includono l'allargamento
dei test sui bordi dei triangoli acuti; la distanza firmata dal piano evita
la cancellazione fra lunghezze nella soglia near. Gli axis array originali
sono condivisi fra CPU e CUDA: ricostruire coordinate con aritmetica diversa
potrebbe cambiare le classificazioni vicine alle soglie.

Il backend CUDA mantiene geometria e assi residenti, classifica batch fino
a262144 candidati interni e salta i nodi FCC dispari prima della traversal.
CUB Flagged conserva l'ordine e restituisce solo boundary compatte, con
staging pinned e aritmetica/memoria/lanci controllati. Non alloca un record
per ogni cella della griglia. I due fence per batch proteggono conteggio e
staging; una futura pipeline doppia deve dimostrare i tempi di riuso dei
buffer. CUDA13 CUB richiede C++17. Il compilatore riporta74 registri Ada e72
GB10 per classifier, senza stack/spill: non sono prestazioni osservate.

La riconciliazione taglia un link da entrambi gli estremi se almeno uno lo
classifica bloccato. Gli endpoint aggiunti diventano rigidi; non propagano
artificialmente un'isolazione completa ad altri nodi. Il controllo finale
richiede bit opposti uguali anche dopo permutazione e folding FCC. Il SAF
somma separatamente ogni link bloccato: due versi opposti contribuiscono2,
correggendo l'addizione booleana del legacy. Kelvin, SAF e near/link reciproci
sono correzioni intenzionali; non si promette bit equality con Python.

`prepare_grid.h` conserva gli8 pesi/corner legacy, permutazioni, parity FCC,
riflessione delle direzioni nel fold e receiver reorder inverso con duplicati.
CPU FCC accetta sia la griglia fisica sia il fold; CUDA usa FCC flag2 ripiegato.
Cinque HDF5 sono pubblicati in una directory esclusivamente nuova e verificati
dal loader originale. PASS75570 check mesh,58485 griglia,140791 preparazione
e7 round trip HDF5, anche ASan/UBSan. I gate CUDA coprono80 scene e code di
batch; il confronto dei record sul device richiede hardware.

Mesh reali con materiali del repository: CTK h=.15m Cart/FCC supera il loader;
Musikverein FCC h=.05m supera geometria/reciprocità/loader con32120 triangoli,
2372701 boundary e128 corner ricevitore. h=.15/.1/.075 per Musikverein è
rifiutato perché i corner di alcuni ricevitori intersecano la mesh: non è un
errore da aggirare. Queste sono prove del port, non frequenze consigliate
o certificazioni della convergenza spaziale delle RIR.

### Aria nativa e sostituzione del costo quadratico modale

`post_air.h` implementa ISO9613, Stokes, OLA e modal. La pressione reale entra
nella concentrazione relativa del vapore e nelle frequenze di rilassamento;
il legacy la fissava alla pressione standard. Stokes usa scatter crescente
nel riferimento CPU e gather per output CUDA, senza atomiche floating point.
OLA usa FFT native radix2/Bluestein sulla CPU e cuFFT FP64 sulla GPU, batch
limitati a128 frame e overlap-add per output con ordine dei frame esplicito.
La finestra legacy dei frame iniziali dist<0 è conservata: a assorbimento
zero il primo tap ha gain7/6 per window1024. Non è un bypass identità e non
va confuso con un nuovo modello di propagazione passivo.

La ricorrenza modal standard costa O(Nin*Nout), parallelizzabile per modo ma
ancora quadratica. `--air-modal-method fft` interpola il Fourier transform
smorzato `Zq(sigma)=sum_n x[n]*exp(-sigma*n-i*pi*q*n/Nout)` aK nodi Chebyshev
Lobatto. I coefficienti della ricorrenza combinano parte reale e immaginaria;
q0 usa direttamente la somma/sqrt(Nout). K FFT di lunghezza2Nout portano il
costo a O(K*Nout*log Nout), evitando una matriceNout*K. La GPU usa batch8 e
somma compensata per modo; una piccola riduzione L1 consente il planning del
bound. I dati del post restano sul device fra DSP e aria.

Il bound conservativo della coda Chebyshev usa
`4*exp(hypot(a,K)-a-K*asinh(K/a))`, con `a=(Nin-1)*sigma_max/2`, e norme L1
e guadagni IDCT per limitare l'errore assoluto d'interpolazione richiesto.
Si mantiene il fallback alla ricorrenza per costo/range/bound sfavorevoli;
la tolleranza non copre roundoff FFT, coefficienti o convergenza fisica.
Il numero di nodi non è fissato arbitrariamente. Questa è un'approssimazione
controllata dell'operatore modale, non una convoluzione stazionaria.

PASS195863 controlli aria/FFT con riferimenti DFT/stato long-double indipendenti
e ASan/UBSan. Otto record lunghi1025..8192 richiedono al massimo25 nodi:
FFTmodal max5.382e-14 contro riferimento long-double, ricorrenza double
max1.521e-10. Sono errori numerici, non speedup. Il modello host dei kernel
gather/batch/spectrum supera18219661 check e sanitizer, rilevando mutanti
con tail/bypass omessi e fase IDCT errata. Il driver CUDA confronta operatori
isolati e pipeline residente, inclusi prime lengths/tail e prefilling NaN.

### HDF5 e WAV della pipeline completa

La CLI mantiene ordine HP/integratore → resample → LP → aria → WAV; legge
Tc/rh dal preparatore, salva atmosfera/metodo/requested tolerance e dimensioni
finali. `--verify` CUDA confronta la pipeline CPU completa. `post_wav.h`
esporta IEEE float32 mono RIFF/fact, normalizzazione globale e native quando
peak<1; il silenzio resta finito. Collisioni, alias, symlink e limiti RIFF32
sono errori espliciti; la pubblicazione esclusiva ha rollback dei propri file.
L'HDF5 è pubblicato prima dei WAV; un errore WAV segnala che l'HDF5 esiste già.
PASS29 casi WAV strict/sanitizer e integrazione dei metadati atmosferici.

Prova reale CTK: preparazione Cart h=.15/Nt1537/Ns8/Nr48, forward CPU FP32,
sei ricevitori postprocessati con OLA48k/LP1400, Stokes8k e modalFFT8k, più
WAV. Il checker indipendente strict/sanitizer verifica227712 campioni HDF5
finiti,9222 raw pesati bit-exact e12WAV/233472 float32 bit-exact. CTK FCC
folded/unfolded Nt1537, CPUFP64: maxdelta5.122e-11 e RMS normalizzato7.80e-15,
PASS per campione1e-10+1e-11*abs(reference). FP32 ha RMS normalizzato3.13e-6
sul raw e1.35e-3 sulla RIR HP/LP filtrata; la cancellazione del Nyquist rende
più visibile il rounding. Il gate FP32 pointwise1e-6+rel1e-3 fallisce e non
viene allargato dopo la prova. FP64 filtrato ha maxdelta9.37e-15 e RMS
normalizzato4.24e-12. Il fold permuta l'ordine delle addizioni: questi risultati
non sostituiscono stabilità lunga, RIR misurate o gate CUDA.
Le istruzioni complete e i mapping dei materiali sono in
[NATIVE_PIPELINE.md](NATIVE_PIPELINE.md). Fitting da nuove assorbenze, temporal
blocking, output asincrono e scelta delle varianti sul device restano lavoro
distinto; non occorre Python per usare la pipeline con i DEF del repository.

### Normalizzazione della sorgente senza NaN

`scale_input` ora gestisce sorgenti tutte nulle con gain identità, preservando
anche signed zero; la simulazione mantiene pressione zero. Conteggi overflow,
storage mancante, NaN/Inf e guadagni non rappresentabili sono errori espliciti.
Le sorgenti normali mantengono esattamente ordine/arithmetic precedenti.
PASS almeno18964 verifiche per precisione,18 errori in processi figli e forwardCPU
Cart/FCC/fold con sorgenti nulle; ASan/UBSan passati. I subnormali che richiedono
un fattore infinito sono rifiutati, senza normalizzazione silenziosa arbitraria.
