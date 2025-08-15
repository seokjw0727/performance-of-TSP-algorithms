### Performance of Travelling Salesman Problem Algorithms

> The objective of this study is to quantitatively assess the execution time and memory utilization of various algorithms on a common benchmark instance.

---

**📁 Repository Structure**
```
performance-of-TSP-algorithms/
├─ main.py          # The main code
├─ gr17.tsp         # Benchmark instance
└─ README.md        # README
```


**⚙️ Environment**
- Python 3.13+ (required venv or conda) and some packages
```
pip install qiskit qiskit-aer tsplib95 psutil
```
- [gr17.tsp](https://exploretsp.netlify.app/tsp/gr17.tsp.gz)


**▶️ Quick Start**
```
python main.py
```
> Tip: Large-scale instances and repeated experiments can require a significant investment of time. Initially, the number of repetitions should be reduced to ascertain the efficacy of the operation, and subsequently, it should be gradually increased.


**🧪 Simulation**
　Quantum algorithms are executed on simulators (Aer) rather than on actual quantum hardware. To address this limitation, experimental interpretation is necessary, taking into account the circuit depth and noise.


**🙌 Acknowledgements**
- [Qiskit](https://qiskit.org/)
- [TSPLIB95](https://www.math.uwaterloo.ca/tsp/tsplib95/)
- [ExploreTSP](https://exploretsp.netlify.app/)

*Any feedback or issues should be directed to the GitHub Issues.*