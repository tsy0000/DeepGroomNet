# Limitations

* Trained on top-view open-field recordings of mice at 30 fps. Other species,
  camera angles, frame rates or arenas require retraining or at least
  recalibration of thresholds.
* The five classes are coarse. Face and body grooming are separated, but
  sub-stages, defecation, jumping and social behaviour are not modelled.
* Behaviour probabilities are frame-level; the model does not estimate
  transition probabilities. SGATF smooths within KPMS syllable runs only and
  must not be interpreted as a learned temporal attention mechanism.
* KPMS syllables are unsupervised clusters; a syllable is not a behaviour
  boundary by definition. SGATF uses run statistics, not semantics.
* Thresholds are cohort-specific. Reusing them on a new camera or strain can
  shift class composition substantially.
* The bundled GUI automates file handling around the same CLIs; it adds no
  modelling capability.
