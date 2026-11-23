// Núcleo en C++ del TFG: test de permutaciones (Monte Carlo) del estudio de eventos.
//
// Hace lo mismo que `permutation_pvalue` y `magnitude_pvalue` de backend/events/study.py, pero
// compilado y repartido entre varios hilos. En cada remuestreo sustituye cada evento real por un
// CAR placebo de la MISMA acción, calcula la media y cuenta cuántas veces es tan extrema como la
// observada. p = (extremas + 1) / (remuestreos + 1).
//
// Reproducibilidad: los remuestreos se agrupan en bloques de BLOCK; cada bloque tiene su propia
// semilla, derivada de (semilla, número de bloque). Así el resultado NO depende de cuántos hilos
// se usen (un hilo o dieciséis dan exactamente el mismo p). El generador es xoshiro256** y el
// entero acotado se saca por rechazo, sin sesgo; ambos son portables entre compiladores.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int64_t BLOCK = 4096;  // remuestreos por bloque (unidad de reparto entre hilos)

inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

// xoshiro256**: generador rápido de buena calidad estadística, inicializado con splitmix64.
struct Rng {
    uint64_t s[4];

    explicit Rng(uint64_t seed) {
        for (auto &x : s) x = splitmix64(seed);
    }

    inline uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Entero uniforme en [0, n) sin sesgo: se descartan los 2^64 mod n valores bajos.
    inline uint64_t below(uint64_t n) {
        const uint64_t limit = (0 - n) % n;
        for (;;) {
            const uint64_t x = next();
            if (x >= limit) return x % n;
        }
    }
};

inline uint64_t block_seed(uint64_t seed, int64_t block) {
    uint64_t state = seed ^ (static_cast<uint64_t>(block) * 0xD1342543DE82EF95ULL);
    return splitmix64(state);
}

struct Problem {
    std::vector<const double *> base;  // base[e]: inicio del conjunto de CAR placebo del evento e
    std::vector<uint64_t> len;         // len[e]: tamaño de ese conjunto
    double observed;
    bool magnitude;
};

// Cuenta cuántos remuestreos del bloque `b` (de `n` remuestreos) son tan extremos como el observado.
int64_t run_block(const Problem &p, uint64_t seed, int64_t b, int64_t n) {
    Rng rng(block_seed(seed, b));
    const size_t events = p.base.size();
    const double target = std::fabs(p.observed);
    int64_t extreme = 0;
    for (int64_t d = 0; d < n; ++d) {
        double sum = 0.0;
        for (size_t e = 0; e < events; ++e) sum += p.base[e][rng.below(p.len[e])];
        const double mean = sum / static_cast<double>(events);
        extreme += p.magnitude ? (mean >= p.observed) : (std::fabs(mean) >= target);
    }
    return extreme;
}

// Contador de un hilo, alineado a 64 bytes (una línea de caché) para que los hilos no se estorben al sumar.
struct alignas(64) Slot {
    int64_t count = 0;
};

unsigned hardware_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1u : n;
}

// pools: conjuntos de CAR placebo (uno por acción); picks[i]: qué conjunto usa el evento i.
// magnitude=false: p de dos colas sobre la media con signo (|media| >= |observed|).
// magnitude=true : p de una cola sobre la media de |CAR| (media >= observed).
double permutation_pvalue(
    const std::vector<std::vector<double>> &pools, const std::vector<int> &picks, double observed,
    int64_t draws, uint64_t seed, bool magnitude, int threads) {
    if (draws < 1) throw std::invalid_argument("draws debe ser al menos 1");
    if (picks.empty()) throw std::invalid_argument("hace falta al menos un evento");
    if (threads < 0) throw std::invalid_argument("threads no puede ser negativo (0 = todos los hilos)");

    // Para la magnitud se trabaja con |CAR|; copiamos solo los conjuntos que algún evento usa.
    std::vector<std::vector<double>> used(pools.size());
    Problem p;
    p.observed = observed;
    p.magnitude = magnitude;
    for (int k : picks) {
        if (k < 0 || static_cast<size_t>(k) >= pools.size())
            throw std::invalid_argument("picks apunta a un conjunto que no existe");
        if (pools[k].empty()) throw std::invalid_argument("un evento usa un conjunto de placebos vacío");
        if (used[k].empty()) {
            used[k] = pools[k];
            if (magnitude)
                for (double &v : used[k]) v = std::fabs(v);
        }
        p.base.push_back(used[k].data());
        p.len.push_back(used[k].size());
    }

    int64_t total = 0;
    {
        py::gil_scoped_release release;  // el cálculo no toca objetos de Python: libera el GIL
        const int64_t blocks = (draws + BLOCK - 1) / BLOCK;
        int64_t nt = threads > 0 ? threads : hardware_threads();
        nt = std::min<int64_t>(nt, blocks);

        std::atomic<int64_t> next_block{0};
        std::vector<Slot> partial(static_cast<size_t>(nt));
        auto worker = [&](size_t slot) {
            for (;;) {
                const int64_t b = next_block.fetch_add(1);
                if (b >= blocks) break;
                const int64_t n = std::min<int64_t>(BLOCK, draws - b * BLOCK);
                partial[slot].count += run_block(p, seed, b, n);
            }
        };
        std::vector<std::thread> pool;
        for (int64_t t = 1; t < nt; ++t) pool.emplace_back(worker, static_cast<size_t>(t));
        worker(0);
        for (auto &t : pool) t.join();
        for (const Slot &slot : partial) total += slot.count;
    }
    return (static_cast<double>(total) + 1.0) / (static_cast<double>(draws) + 1.0);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Núcleo C++ del TFG: test de permutaciones multihilo para el estudio de eventos.";
    m.def(
        "permutation_pvalue", &permutation_pvalue, py::arg("pools"), py::arg("picks"), py::arg("observed"),
        py::arg("draws"), py::arg("seed"), py::arg("magnitude") = false, py::arg("threads") = 0,
        "p-valor de permutaciones. pools: lista de listas de CAR placebo; picks: índice de lista por evento.");
    m.def("max_threads", []() { return hardware_threads(); }, "Hilos de hardware disponibles.");
    m.attr("BLOCK") = BLOCK;
}
