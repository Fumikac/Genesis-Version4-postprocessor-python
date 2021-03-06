# Genesis 1.3 Virson 4 post processor in Python

Fumika Isono

Post-processor (Python) of [Genesis 1.3 Version 4](https://github.com/svenreiche/Genesis-1.3-Version4), Time-dependent, 3D Code to simulate the amplification process of a Free-electron Laser.

- Can calculate radiation frequency (far-field and near-field) for Time-dependent case.
- Includes postprocessor [GenesisOut.py](functions/GenesisOut.py) of the output file .out.h5, and postprocessor [ParticleOut.py](functions/ParticleDump.py) for particle dump files (.par.h5). With the code, you can track all simulated particles.

## Example cases
Benchmarks provided by the author of Genesis 1.3 Version 4
- [results](result-Benchmark1-SASE.ipynb) of [Benchmark1-SASE](https://github.com/svenreiche/Genesis-1.3-Version4/tree/master/benchmark/Benchmark1-SASE)
- [results](result-Benchmark4-UndErrors.ipynb) of [Benchmark4-UndErrors](https://github.com/svenreiche/Genesis-1.3-Version4/tree/master/benchmark/Benchmark4-UndErrors)

VISA undulator, Time-dependent simulation
- [results](result-VISA-Gaussian-Seed.ipynb) of [VISA-Gausssian-Seed](benchmark/VISA-Gaussian-Seed)
