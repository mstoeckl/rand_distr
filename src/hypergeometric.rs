//! The hypergeometric distribution `Hypergeometric(N, K, n)`.
#![allow(non_snake_case)]
use crate::Distribution;
use core::{f64, fmt};
#[allow(unused_imports)]
use num_traits::Float;
use rand::distr::uniform::Uniform;
use rand::{Rng, RngExt};

/// The [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution) `Hypergeometric(N, K, n)`.
///
/// This is the distribution of successes in samples of size `n` drawn without
/// replacement from a population of size `N` containing `K` success states.
///
/// See the [binomial distribution](crate::Binomial) for the analogous distribution
/// for sampling with replacement. It is a good approximation when the population
/// size is much larger than the sample size.
///
/// # Density function
///
/// `f(k) = binomial(K, k) * binomial(N-K, n-k) / binomial(N, n)`,
/// where `binomial(a, b) = a! / (b! * (a - b)!)`.
///
/// # Plot
///
/// The following plot of the hypergeometric distribution illustrates the probability of drawing
/// `k` successes in `n = 10` draws from a population of `N = 50` items, of which either `K = 12`
/// or `K = 35` are successes.
///
/// ![Hypergeometric distribution](https://raw.githubusercontent.com/rust-random/charts/main/charts/hypergeometric.svg)
///
/// # Example
/// ```
/// use rand_distr::{Distribution, Hypergeometric};
///
/// let hypergeo = Hypergeometric::new(60, 24, 7).unwrap();
/// let v = hypergeo.sample(&mut rand::rng());
/// println!("{} is from a hypergeometric distribution", v);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Hypergeometric {
    /* number of samples */
    n: u64,
    /* population size */
    N: u64,
    /* number with feature */
    K: u64,
}

/// Error type returned from [`Hypergeometric::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `total_population_size` is zero
    ZeroPopulation,
    /// `population_with_feature > total_population_size`.
    ProbabilityTooLarge,
    /// `sample_size > total_population_size`.
    SampleSizeTooLarge,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Error::ZeroPopulation => "total_population_size is zero",
            Error::ProbabilityTooLarge => {
                "population_with_feature > total_population_size in geometric distribution"
            }
            Error::SampleSizeTooLarge => {
                "sample_size > total_population_size in geometric distribution"
            }
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl Hypergeometric {
    /// Constructs a new `Hypergeometric` with the shape parameters
    /// `N = total_population_size`,
    /// `K = population_with_feature`,
    /// `n = sample_size`.
    #[allow(clippy::many_single_char_names)] // Same names as in the reference.
    pub fn new(
        total_population_size: u64,
        population_with_feature: u64,
        sample_size: u64,
    ) -> Result<Self, Error> {
        if total_population_size == 0 {
            return Err(Error::ZeroPopulation);
        }

        if population_with_feature > total_population_size {
            return Err(Error::ProbabilityTooLarge);
        }

        if sample_size > total_population_size {
            return Err(Error::SampleSizeTooLarge);
        }

        Ok(Hypergeometric {
            n: sample_size,
            N: total_population_size,
            K: population_with_feature,
        })
    }
}

/// Estimate `x log_2 x` as a 72.56 fixed point number. The last few bits
/// will be incorrect.
fn xlog2x_accurate(x: u64) -> u128 {
    if x == 0 {
        /* lim_{x->0} x log x = 0 */
        return 0;
    }

    let mut out = 0u128;

    /* Iterative binary logarithm calculation; very slow, but doesn't require
     * many constants and is easily specialized to any width */
    const TAIL: u32 = 56;

    out = out.strict_add((u128::from(x) * u128::from(x.ilog2())).strict_mul(1 << TAIL));

    let mut os = TAIL as i32 - 1; // output shift

    let mut y = u128::from(x) << (127 - x.ilog2()); // 1.127 fixed point with high bit set
    assert!(y >> 127 == 0x1);

    while os > -128 {
        /* y -> y^2 or (y^2)/2 */
        let (l, h) = y.carrying_mul(y, 0);
        if h >> 127 == 0x1 {
            /* y^2 is in [2,4), do out += (x << os) */
            out = out.strict_add(if os >= 0 {
                u128::from(x) << os as u32
            } else {
                u128::from(x) >> (-os) as u32
            });
            y = h;
        } else {
            /* y^2 still in range. The low bit may be safe to discard/has limited error. */
            y = (h << 1) | (l >> 127);
        }
        assert!(y >> 127 == 0x1);
        os -= 1;
    }

    out
}

/// Estimate 2^{-x} for x in 0.128 fixed point value; outputs a 1.127 fixed point value
fn exp2n(x: u128) -> u128 {
    /* Note: some additional range reduction and precision improvements can be
     * implemented using additional tables, and other algebraic tricks. But these
     * make the design rather more complicated. */

    /* Step 1: range reduction & rescaling, y = x/2^12 */
    let mult_d128 = |x: u128, y: u128| x.carrying_mul(y, 0).1;
    const LN2: u128 = 0xb17217f7d1cf79abc9e3b39803f2f6af;
    let y = mult_d128(x >> 12, LN2);

    /* Step 2: taylor, z = 1 - y + y^2/2 - .. + y^8/8!, yields roughly 2^{-110} precision,
     * but some of it is lost later when undoing the range reduction */
    let y2 = mult_d128(y, y);
    let y3 = mult_d128(y2, y);
    let y4 = mult_d128(y2, y2);
    let y5 = mult_d128(y4, y);
    let y6 = mult_d128(y4, y2);
    let y7 = mult_d128(y6, y);
    let y8 = mult_d128(y4, y4);

    /* Note: because the divisor is small, directly implementing the
     * divisions might actually be more efficient. */
    const D3F: u128 = 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;
    const D4F: u128 = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;
    const D5F: u128 = 0x2222222222222222222222222222222;
    const D6F: u128 = 0x5b05b05b05b05b05b05b05b05b05b0;
    const D7F: u128 = 0xd00d00d00d00d00d00d00d00d00d0;
    const D8F: u128 = 0x1a01a01a01a01a01a01a01a01a01a;

    let mut dz = y;
    dz -= y2 >> 1;
    dz += mult_d128(y3, D3F);
    dz -= mult_d128(y4, D4F);
    dz += mult_d128(y5, D5F);
    dz -= mult_d128(y6, D6F);
    dz += mult_d128(y7, D7F);
    dz -= mult_d128(y8, D8F);
    let z = (1 << 127) - (dz >> 1);

    /* Step 3: w -> w^(2^12) */

    let mult_d127 = |x: u128, y: u128| {
        let (lo, hi) = x.carrying_mul(y, 0);
        (hi << 1) | (lo >> 127)
    };
    let mut w = z;
    for _ in 0..12 {
        w = mult_d127(w, w);
    }

    let fast = (((-(x as f64) / (2.0f64.powi(128))).exp2()) * 2.0.powi(127) as f64).round() as u128;
    assert!(w.abs_diff(fast) < 1 << 80);
    w
}

/// Update estimate `v` (2.126) of `log_2 z` for `z` in [1,2) (as 1.127 fixed point)
fn log2x_newton_step(v: u128, z: u128) -> u128 {
    /* v <- v + 1/(ln 2) z 2^{-v} - 1/(ln 2)
     *
     * For v in [0,1], this may produce intermediates up in [0, 4),
     */
    let log2e = 0xb8aa3b295c17f0bbbe87fed0691d3e89u128; // 1.44... in 1.127
    let log2e_126 = log2e >> 1;

    assert!(v <= 1 << 126);

    let e2n = if v == 1 << 126 {
        1 << 126
    } else {
        exp2n(v << 2)
    };
    let (p_lo, p_hi) = e2n.carrying_mul(z, 0); // 0.128 * 1.127 -> 1.127
    let p = (p_hi << 1) | (p_lo >> 127); // 1.127
    let (_q_lo, q_hi) = p.carrying_mul(log2e, 0);
    let q = q_hi; // 2.126

    let out = v.strict_add(q).strict_sub(log2e_126);
    /* Clip to [0,1), because log_2(x) for x in [1,2) is in [0,1) */
    out.min(1 << 126)
}

// Estimate `log_2 x` for `x` in [1,2) (as 2.126 fixed point) as a 0.128 fixed point value
fn log2x_core(x: u128) -> u128 {
    /* `as` saturates */
    let v0 = ((((x as f64) / (2.0f64.powi(127))).log2()) * 2.0.powi(126) as f64).round() as u128;

    let v1 = log2x_newton_step(v0, x);

    v1
}

/// Estimate `log_2 x` as a 8.120 fixed point number. The last few bits
/// will be incorrect.
fn log2x(x: u64) -> u128 {
    assert!(x > 0);

    let tail = u128::from(x.ilog2()).strict_mul(1 << 120);
    let y = u128::from(x) << (127 - x.ilog2());
    let yc = log2x_core(y);
    let out = tail.strict_add(yc >> 6);

    out
}

/// Estimate `x log_2 x` as a 72.56 fixed point number. The last few bits
/// will be incorrect.
fn xlog2x_medium(x: u64) -> u128 {
    if x == 0 {
        /* lim_{x->0} x log x = 0 */
        return 0;
    }

    let l2 = log2x(x);
    let (out_lo, out_hi) = u128::from(x).carrying_mul(l2, 0);
    (out_lo >> 64) | (out_hi << 64)
}

/// Estimate `x log_2 x` as a 72.56 fixed point number. This only ensures relative
/// 2^{-53} accuracy
fn xlog2x_fast(x: u64) -> u128 {
    if x == 0 {
        /* lim_{x->0} x log x = 0 */
        return 0;
    }

    (((x as f64) * (x as f64).log2()) * (1u64 << 56) as f64).round() as u128
}

/// Estimate `x log_2 x` as a 72.56 fixed point number. Balances speed and accuracy.
fn xlog2x(x: u64) -> u128 {
    let val = if x <= (1 << 16) {
        xlog2x_fast(x)
    } else {
        xlog2x_medium(x)
    };
    #[cfg(fuzzing)]
    {
        // Probabilities should be calculated with 24 bits of accuracy
        let acc = xlog2x_accurate(x);
        assert!(
            acc.abs_diff(val) <= 1 << (56 - 24),
            "{} {:032x} {:032x}",
            x,
            acc,
            val
        );
    }

    val
}

/// Estimate `x log_2 e` as a 72.56 fixed point number. The last few bits
/// will be incorrect.
fn xlog2e(x: u64) -> u128 {
    let x = u128::from(x);
    /* sympy.N((1<<127) / sympy.log(2), n=39) */
    let log2e = 0xb8aa3b295c17f0bbbe87fed0691d3e89u128;

    /* 128.0 * 1.127 => 72.56 */
    let (l, h) = x.carrying_mul(log2e, 0);
    (h << 57) | (l >> (128 - 57))
}

/// Estimates: `log_2(Gamma(x + 1)) - (x log2(x) - x log2(e))`
///
/// Requires x >= 1
fn log2_factorial_residual(x: u64) -> f64 {
    /* Table lookup for small entries, where Stirling is inaccurate.
     * TABLE[i] = log2(x!) - (x log2(x) - x log2(e))
     */
    #[allow(clippy::approx_constant)]
    const TABLE: [f64; 16] = [
        0.0,
        1.4426950408889634,
        1.8853900817779268,
        2.1581601212245776,
        2.3557426642770096,
        2.5107253256165234,
        2.6382483373365173,
        2.7465888502067934,
        2.8407683454989856,
        2.92406337484945,
        2.998730574732963,
        3.066390378122578,
        3.128245716089093,
        3.1852141479388245,
        3.238011537913132,
        3.287207149089294,
    ];
    if x < TABLE.len() as u64 {
        return TABLE[x as usize];
    }

    // Euler–Maclaurin variant of Stirling's approximation; with
    // x >= 16 needing about n=13 terms to get on the order of
    // f64::EPSILON precision.
    let xf = x as f64;
    let xf3 = xf.powi(3);
    let xf5 = xf.powi(5);
    let xf7 = xf.powi(7);
    let xf9 = xf.powi(9);
    let xf11 = xf.powi(11);
    let xf13 = xf.powi(13);

    let resid_tail = 1.0 / (12.0 * xf) - 1.0 / (360.0 * xf3) + 1.0 / (1260.0 * xf5)
        - 1.0 / (1680.0 * xf7)
        + 1.0 / (1188.0 * xf9)
        - 691.0 / (360360.0 * xf11)
        + 1.0 / (156.0 * xf13);

    0.5 * (f64::consts::TAU * xf).log2() + f64::consts::E.log2() * resid_tail
}

/// log2(x!) as sum of 72.56 fixed point and f64 float
fn log2_factorial_parts(x: u64) -> (i128, f64) {
    /* Note: at x = (1,2), `x log2(e) >= x log2(x)` */
    let m = i128::try_from(xlog2x(x))
        .expect("bounded")
        .strict_sub(i128::try_from(xlog2e(x)).expect("bounded"));
    (m, log2_factorial_residual(x))
}

/// log2 of the hypergeometric probability mass function. Output will always be nonpositive.
fn log_pmf_brute(N: u64, K: u64, n: u64, k: u64) -> f64 {
    assert!(k <= n && n <= N && K <= N);
    if k > K || k < n.saturating_sub(N.strict_sub(K)) {
        return -f64::INFINITY;
    }

    /* ⎛K⎞ ⎛N-K⎞   ⎛N⎞   K! (N-K)! n! (N-n)!
     * ⎝k⎠ ⎝n-k⎠ / ⎝n⎠ = ------------------------------
     *                   k! (K-k)! (n-k)! (N-K+k-n)! N!
     */

    let denom: [(i128, f64); _] = [
        log2_factorial_parts(k),
        log2_factorial_parts(K.strict_sub(k)),
        log2_factorial_parts(n.strict_sub(k)),
        log2_factorial_parts(N.strict_sub(K).strict_add(k).strict_sub(n)),
        log2_factorial_parts(N),
    ];
    let num: [(i128, f64); _] = [
        log2_factorial_parts(K),
        log2_factorial_parts(N.strict_sub(K)),
        log2_factorial_parts(n),
        log2_factorial_parts(N.strict_sub(n)),
    ];
    let mut main = 0i128;
    let mut resid = 0.0f64;
    for (m, r) in denom {
        main = main.strict_add(m);
        resid += r;
    }
    for (m, r) in num {
        // logically this should remain positive, but floating point error exists
        main = main.strict_sub(m);
        resid -= r;
    }

    let val = -((main as f64) * (1.0 / (1u64 << 56) as f64) + resid);
    debug_assert!(
        val <= 1e-10,
        "(N,K,n,k)=({},{},{},{}) {}=(-{}-{})",
        N,
        K,
        n,
        k,
        val,
        (main as f64) * (1.0 / (1u64 << 56) as f64),
        resid
    );

    val.min(0.0)
}

#[test]
fn test_hg() {
    use std::println;

    for v in [
        1,
        2,
        3,
        15,
        16,
        300,
        100000,
        1 << 10,
        1 << 20,
        (1 << 30) - 1,
        (1 << 60) + 1,
        (1 << 63) - (1 << 33) + (1 << 10) - 7,
    ] {
        let fast = xlog2x_fast(v);
        let medium = xlog2x_medium(v);
        let acc = xlog2x_accurate(v);
        println!(
            "xlog2x({}): accurate {:032x} med {:032x} ({:.3}) fast {:032x} ({:.3})",
            v,
            acc,
            medium,
            (acc.abs_diff(medium) as f64).log2(),
            fast,
            (acc.abs_diff(fast) as f64).log2()
        );
        assert!(acc.abs_diff(medium) < 1 << 16);

        log2_factorial_residual(v);
    }

    for (N, K, n, k) in [
        (500, 400, 30, 20),
        (250, 200, 230, 200),
        (100, 20, 6, 3),
        (50, 10, 47, 35),
        (5000, 2500, 500, 200),
        (10100, 10000, 1000, 50),
        (100100, 100, 10000, 50),
        (1 << 63, 100, 10000, 1),
    ] {
        println!("NKnk {} {} {} {}", N, K, n, k);
        log_pmf_brute(N, K, n, k);
    }
}

/// ⌈σ⌉, where σ is the standard deviation of Hypergeo(N,K,n),
/// whose variance is:
///
/// ```text
///      n K (N-K) (N-n)
/// σ² = ----------------
///       N N (N - 1)
/// ```
fn hypergeo_ceil_sqrt_variance(N: u64, K: u64, n: u64) -> u64 {
    let mut tmp = [0u64; 4];

    let mul = |t: &mut [u64; 4], v: u64| {
        let mut carry = 0;
        for x in t.iter_mut() {
            let (nx, ncarry) = x.carrying_mul(v, carry);
            *x = nx;
            carry = ncarry;
        }
        assert!(carry == 0);
    };
    let add = |t: &mut [u64; 4], mut v: u64| {
        for x in t.iter_mut() {
            let (nx, o) = x.overflowing_add(v);
            *x = nx;
            if !o {
                return;
            }
            v = 1;
        }
        unreachable!(); // addition doesn't overflow'
    };
    let div = |t: &mut [u64; 4], v: u64| {
        /* Chunked long division */
        let mut out1 = [0u64; 4];
        let mut out2 = [0u64; 4];
        for i in (0..(t.len() - 1)).rev() {
            let chunk = (u128::from(t[i + 1]) << 64) + u128::from(t[i]);
            let leftover = chunk % u128::from(v);
            /* unnecessary: t[i+1] = 0; */
            t[i] = u64::try_from(leftover).expect("becaues v <= u64::MAX");
            let div = chunk / u128::from(v);
            let div_hi = (div >> 64) as u64;
            let div_lo = div as u64;
            out2[i + 1] = div_hi;
            out1[i] = div_lo;
        }

        let mut carry = false;
        for ((o1, o2), x) in (out1.iter_mut().zip(out2.iter_mut())).zip(t.iter_mut()) {
            let (nx, ncarry) = o1.carrying_add(*o2, carry);
            *x = nx;
            carry = ncarry;
        }
        assert!(!carry);
    };
    let div_ceil = |t: &mut [u64; 4], v: u64| {
        add(t, v.strict_sub(1));
        div(t, v);
    };

    add(&mut tmp, n);
    mul(&mut tmp, K);
    mul(&mut tmp, N.strict_sub(K));
    mul(&mut tmp, N.strict_sub(n));
    div_ceil(&mut tmp, N);
    div_ceil(&mut tmp, N);
    div_ceil(&mut tmp, N.strict_sub(1));
    assert!(tmp[1] == 0 && tmp[2] == 0 && tmp[3] == 0);
    let ceil_variance = tmp[0];

    let s = ceil_variance.isqrt();
    if s * s == ceil_variance { s } else { s + 1 }
}

impl Distribution<u64> for Hypergeometric {
    #[allow(clippy::many_single_char_names)] // Same names as in the reference.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.n == 0 || self.K == 0 {
            return 0;
        }
        if self.K == self.N {
            return self.n;
        }
        if self.n == self.N {
            return self.K;
        }

        /* Standard rejection sampling method. Distribution indicated
         * by dots. Blocks indicate the bounding distribution (chosen
         * to be relatively easy to sample and prove correctness.)
         *
         *
         *                CC.CCCCC
         *                CC.CCCCC
         *                CC..CCCC
         *           lllllC....CCCrrrrr
         *           lllll......CCrrrrr
         *           lll..........rrrrr
         * LLLLLLLLL.................rrRRRRRRRRRR
         *
         * This sampler is efficient if the sum of block areas is O(1).
         *
         * Central section: follows from anticoncentration (hopefully,
         * assuming hypergeometric distribution is be close enough to
         * binomial.)
         *
         * Tail section: follows from variant of Chebyshev's and Gauss's
         * inequalities, requires distribution be unimodal.
         */

        let ceil_std = hypergeo_ceil_sqrt_variance(self.N, self.K, self.n);
        assert!(ceil_std >= 1);

        /* Calculate integer quantities exactly, to avoid sampling errors or tricky bugs if
         * the central block or tail probabilities are wrong */
        let mode_denom = u128::from(self.N).strict_add(2);
        let mode_num = (u128::from(self.n).strict_add(1)) * (u128::from(self.K).strict_add(1));
        let mode_lo = u64::try_from(
            mode_num
                .div_ceil(mode_denom)
                .saturating_sub(1)
                .min(u128::from(self.n)),
        )
        .expect("<= n");
        let mode_hi = u64::try_from((mode_num / mode_denom).min(u128::from(self.n))).expect("<= n");

        let mean_lo = u64::try_from((u128::from(self.n) * u128::from(self.K)) / u128::from(self.N))
            .expect("<= n");
        let mean_hi =
            u64::try_from((u128::from(self.n) * u128::from(self.K)).div_ceil(u128::from(self.N)))
                .expect("<= n");
        assert!(mean_hi <= self.n);

        /* For a unimodal distribution with mean μ, mode `m`, and variance σ^2:
         *
         * for all k >= max(μ, m)
         *
         * σ^2 =  sum_{j} p(j) (j-μ)^2                          (definition)
         *     >= sum_{j=max(μ, m)}^{k} p(j) (j-μ)^2
         *     >= p(k) sum_{j=max(μ, m)}^{k} (j-max(μ, m))^2    (monotonicity!)
         *      = p(k) sum_{h=0}^{k - max(μ, m)} h^2
         *     >= p(k) (k - max(μ, m))^3 / 3
         *
         * Let b = ⌈σ⌉ and t >= 1; then
         *
         * p(max(μ, m) + b t) <= 3 σ^2 / (b t)^3
         *                    <= 3 / b t^3
         *
         * In particular, considering the block of `k` in
         *      [max(μ, m) + 2^l b, max(μ, m) + 2^{l+1} b),
         * the maximum probability of p in the block is
         *      <= 3 / (b 2^{3 l})
         * which multiplied by block width b 2^l is:
         *      <= 3 / (2^{2 l})
         */

        let tbsize = ceil_std;

        /* Note: `mode_ideal - mean_ideal` should be <= 1 in practice, so this
         * only makes a slight difference in the central block range. */
        let cblock_lo = mode_lo.min(mean_lo).saturating_sub(tbsize - 1);
        let cblock_hi = mode_hi.max(mean_hi).saturating_add(tbsize - 1);

        let val1 = log_pmf_brute(self.N, self.K, self.n, mode_lo);
        let val2 = log_pmf_brute(self.N, self.K, self.n, mode_hi);

        let modal_prob = val1.min(val2).exp2();

        let cblock_total_prob = modal_prob * (cblock_hi - cblock_lo + 1) as f64;
        /*  Not proven, but only affects efficiency. Empirical bound w/ high precision is actually 2.0. */
        debug_assert!(cblock_total_prob <= 4.0, "for the fuzziest");
        /* sum_{l=0}^{inf} 1/4^l = 4/3 */
        let wing_max_prob = 4.0;
        let side_base_prob = 3.0 / tbsize as f64;

        let max_prob = 2.0 * wing_max_prob + cblock_total_prob;

        /* [wing L] [wing R] [center] */
        let segment_dist = Uniform::new(0., max_prob).unwrap();
        let rej_dist = Uniform::new(0.0, 1.0).unwrap();
        loop {
            let v = rng.sample(segment_dist);
            let (k, p_upper) = if v <= wing_max_prob {
                /* Left */
                let l = (-(v / wing_max_prob).log2()).floor() as u32 / 2;
                /* Note: p(l) = i = 3/4^i */
                let Some(tl) = 1u64.checked_shl(l) else {
                    continue;
                };
                let Some(bw) = tl.checked_mul(tbsize) else {
                    /* overflow implies block is outside of 0..u64::MAX */
                    continue;
                };
                let Some(block_hi) = mode_lo.min(mean_lo).checked_sub(bw) else {
                    continue;
                };
                let d = rng.sample(Uniform::new(0, bw).unwrap());
                let Some(k) = block_hi.checked_sub(d) else {
                    continue;
                };
                (k, side_base_prob / (tl as f64).powi(3))
            } else if v <= 2.0 * wing_max_prob {
                /* Right */
                let l = (-(((v - wing_max_prob) / wing_max_prob).log2())).floor() as u32 / 2;
                // println!("{} {}", ((v - wing_max_prob) / wing_max_prob), l);
                let Some(tl) = 1u64.checked_shl(l) else {
                    continue;
                };
                let Some(bw) = tl.checked_mul(tbsize) else {
                    /* overflow implies block would be outside of 0..u64::MAX */
                    continue;
                };
                let Some(block_lo) = mode_hi.max(mean_hi).checked_add(bw) else {
                    continue;
                };
                let d = rng.sample(Uniform::new(0, bw).unwrap());
                let Some(k) = block_lo.checked_add(d) else {
                    continue;
                };
                (k, side_base_prob / (tl as f64).powi(3))
            } else {
                /* Sample center */
                let k = rng.sample(Uniform::new_inclusive(cblock_lo, cblock_hi).unwrap());
                (k, modal_prob)
            };
            if k > self.n {
                continue;
            }

            let p_actual = log_pmf_brute(self.N, self.K, self.n, k).exp2();
            let y = rej_dist.sample(rng);
            // TODO: tune and check with fuzzer after calculations are made more precise.
            /* When the peak is broad, this may get tripped by pmf calculation error
             * when values other than the actual mode have higher calculated probability. */
            debug_assert!(
                p_actual <= p_upper * (1.1),
                "calculation accuracy {} <?≈ {} for (N,K,n,k)=({},{},{},{})",
                p_actual,
                p_upper,
                self.N,
                self.K,
                self.n,
                k
            );
            /* This always skips if p_actual = 0.0 */
            if p_upper * y < p_actual {
                return k;
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_hypergeometric_invalid_params() {
        assert!(Hypergeometric::new(100, 101, 5).is_err());
        assert!(Hypergeometric::new(100, 10, 101).is_err());
        assert!(Hypergeometric::new(100, 101, 101).is_err());
        assert!(Hypergeometric::new(100, 10, 5).is_ok());
    }

    fn test_hypergeometric_mean_and_variance<R: Rng>(n: u64, k: u64, s: u64, rng: &mut R) {
        let distr = Hypergeometric::new(n, k, s).unwrap();

        let expected_mean = s as f64 * k as f64 / n as f64;
        let expected_variance = {
            let numerator = (s * k * (n - k) * (n - s)) as f64;
            let denominator = (n * n * (n - 1)) as f64;
            numerator / denominator
        };

        let mut results = [0.0; 1000];
        for i in results.iter_mut() {
            *i = distr.sample(rng) as f64;
        }

        let mean = results.iter().sum::<f64>() / results.len() as f64;
        assert!(
            (mean - expected_mean).abs() < expected_mean / 50.0,
            "|µ-Μ| ≥ M/50 with µ={}, Μ={} at (N,K,n)=({},{},{})",
            mean,
            expected_mean,
            n,
            k,
            s
        );

        let variance =
            results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / results.len() as f64;
        assert!(
            (variance - expected_variance).abs() < expected_variance / 10.0,
            "|σ^2-V| ≥ V/50 with σ^2={}, V={} at (N,K,n)=({},{},{})",
            variance,
            expected_variance,
            n,
            k,
            s
        );
    }

    #[test]
    fn test_hypergeometric() {
        let mut rng = crate::test::rng(737);

        // exercise algorithm HIN:
        test_hypergeometric_mean_and_variance(500, 400, 30, &mut rng);
        test_hypergeometric_mean_and_variance(250, 200, 230, &mut rng);
        test_hypergeometric_mean_and_variance(100, 20, 6, &mut rng);
        test_hypergeometric_mean_and_variance(50, 10, 47, &mut rng);

        // exercise algorithm H2PE
        test_hypergeometric_mean_and_variance(5000, 2500, 500, &mut rng);
        test_hypergeometric_mean_and_variance(10100, 10000, 1000, &mut rng);
        test_hypergeometric_mean_and_variance(100100, 100, 10000, &mut rng);
    }

    #[test]
    fn hypergeometric_distributions_can_be_compared() {
        assert_eq!(Hypergeometric::new(1, 2, 3), Hypergeometric::new(1, 2, 3));
    }
}
