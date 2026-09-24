# Step 2 — Planet runtime migration

## Added / canonicalized

- robust MSS screen capture with exact monotonic acquisition timing and lazy BGRA -> RGB conversion
- Win32 Planet client-window discovery and foreground safety gate
- canonical player/enemy YOLO adapters
- canonical platform segmentation adapter using the existing LightUNet checkpoint contract
- platform probability-map -> surface extraction
- composed perception backend that runs all detectors on the same frame
- read-only `PlanetLiveObserver`
- runtime smoke test CLI
- migration `.cmd` wrapper that copies only selected model weights
- runtime regression tests

## Deliberately not migrated

- old training scripts
- V1/V2/final/structured policy implementations
- old smoke/preview scripts
- generated training directories
- legacy low-level policy code into production `src/`

The existing clean input controller was already canonical and remains unchanged.
