# Low-precision Phase Only Beamforming: When Cheap Hardware Meets Smart Algorithms [with Python code]

Low-precision Phase Only Beamforming: When cheap hardware meets intelligent algorithms, achieve nearly ideal array signal processing performance with the lowest precision phase shifter. In large-scale antenna array systems, traditional digital beamforming requires high-resolution analog-to-digital converters and high-precision phase shifters for each element, leading to a linear increase in hardware cost and power consumption with the array size. This paper systematically studies a low-precision phase-only beamforming technique, combining the Slepian subspace dimensionality reduction method with discrete phase optimization. It demonstrates that even under 1-bit phase quantization conditions, only 2 to 3 measurements are needed to approximate the theoretical upper limit of continuous-precision MPDR beamformers. Experiments show that the proposed method significantly outperforms the classical discrete single-mode MPDR method in terms of interference suppression depth, array gain, and robustness. It provides a new theoretical basis and practical solution for low-cost large-scale MIMO front-end design.
Keywords: Beamforming; Phase-only control; Low precision quantization; Slepian subspace; MPDR
I. Introduction 1.1 Research background The demand for large-scale antenna arrays in the fifth and sixth generation mobile communication systems is becoming increasingly urgent. In traditional all-digital beamforming architectures, each antenna requires an independent RF link, including a high-precision analog-to-digital converter and a tunable phase shifter. When the number of elements reaches hundreds or even thousands, the hardware power consumption, cost and heat dissipation pressure rise sharply, becoming a key bottleneck restricting the system deployment.
The phase-only control architecture is a highly attractive compromise: each antenna element is controlled by a phase shifter, with amplitudes kept constant, thus significantly simplifying the RF frontend. However, in practice, phase shifters can only operate within limited precision—typical devices offer resolutions of 4 to 6 bits, and even lower down to 1 to 2 bits when cost reduction is pursued. The key question is: under extremely low phase accuracy constraints, can we still design beamformers that perform close to ideal values?
1.2 Problem definition Consider a uniform linear array with M elements, where the array receives signals modeled as:
The following is the translation of the Chinese text into English:
Y = a_s · s + a_i · i + n, where a_s and a_i are the steering vectors for signals and interference, s and i are the complex amplitudes of signals and interference, and n is the complex Gaussian noise.
The goal of the MPDR beamformer is to minimize the total output power under the constraint of ensuring undistorted target direction gain:
min_w wH·R·w, subject to wH·a_s = 1. The closed-form solution is w_opt∝R⁻¹·a_s, which is the classic Capon beamformer.
When phase control only (|w_m| = 1/√M) and bit-quantized phase (phase values limited to 2^b discrete values) are added, the problem becomes an NP-hard combinatorial optimization problem. Traditional methods usually adopt naive quantization: first calculate the continuous MPDR solution, then round its phase to the nearest discrete grid. This method performs reasonably well in high precision but degrades severely when b ≤ 2.
1.3 Core Innovation The core idea of this paper comes from the "Low Bit-Precision Dedimensionalization Broadband Beamforming" framework published by DeLude et al. in IEEE JSTSP 2025, and it is narrowed down and monomoded. The core innovation includes three points:
1. Measurement matrix design replaces beamformer design: Instead of directly optimizing the beamforming weight vector w, a K×M measurement matrix Φ (K << M) is designed, with each row being a b-bit monomodal quantized vector. The array output is compressed through Φ before digital reconstruction.
2. Monotonicity-guaranteed recursive design algorithm: The design_monotone method is proposed, which ensures that the estimated variance decreases monotonically with the number of bits through a hot start from the b-1-bit solution combined with element-wise coordinate optimization, eliminating the performance regression phenomenon of the naive greedy algorithm when b ≥ 2.
3.
Multi-measurement vector synergy gain: Classical single-mode MPDR operates only under the setting of K=1 (one weight vector corresponds to one measurement). This paper demonstrates that by jointly designing K ≥ 2 low-bit measurement vectors, interference suppression performance can be significantly improved without increasing the single-path accuracy.
II. Method 2.1 Signal Model and Slepian Beamforming In the Slepian framework, the original M-dimensional array output y is first compressed to K-dimensionality through the measurement matrix Φ:
w_meas = Φ · y = Φ · (a_s·s + a_i·i + n) Reconstruction phase: weighted least squares estimate of signal amplitude.
α̂ = (ΨH·W·Ψ + δ·I)⁻¹ · ΨH·W·w_meas where Ψ = Φ·A_θ is the projected steering matrix after measurement matrix projection, W is the weighting matrix. For MPDR, W takes the inverse of the reduced-dimension covariance matrix.
The equivalent weight vectors reconstructed are:
w_effH = (ΨH·R_w⁻¹·Ψ)⁻¹ · Ψ^H·R_w⁻¹·Φ
2.2 Design of measurement matrix for MPDR objective, setting the target measurement column vector φ_star = R⁻¹·a_s. The method to randomly generate candidate measurement rows is:
v = φ_star · β, where β ~ CN(0, I_D) and subsequently perform bit-level phase quantization on each element of v to obtain candidate single-mode vectors. Among J random candidates, the one with the smallest estimated posterior variance is chosen to be added to Φ.
The variance calculation formula is:
Var = tr[(ΨH·(Φ·R·ΦH)⁻¹·Ψ)⁻¹] This formula takes into account both the structure of the measurement matrix and the impact of the noise/interference covariance matrix, making it the most direct and rigorous selection criterion.
2.3 Monotonicity Guarantee through Recursive Design
The plain greedy design faces a critical failure mode: the variance of the b=2 measurement matrix design may be higher than that of the b=1 design, due to the fact that while the feasible set for b=2 includes the feasible set for b=1 (any vector of b=1 is also a valid b=2 vector), the random candidate generator almost never samples patterns similar to b=1. Once a greedy algorithm converges locally optimally, it cannot "discover" low-bit solutions.
The design_monotone algorithm addresses this issue by employing the following recursive structure:
1. Base case (b=1): Initially design Φ using standard greedy search and perform element-wise coordinate optimization.
2. Recursive step (b→b+1): Start with the b+1 bit optimal solution as the initial solution for b+1 bit, and then perform element-wise coordinate optimization on this basis—try all 2^(b+1) discrete phase values for each element of Φ, retaining the choice that minimizes variance.
Since element-wise coordinate optimization can traverse all single-element variations, and the b+1 bit solution ensures variance does not increase within the b+1 bit space, it strictly guarantees Var(b+1) ≤ Var(b).
III. Experimental Design
3.1 Scene Setup Parameters
- Element Number M
- 32
- Spacing Between Elements
- λ/2
- Signal Direction
- 5°
- Interference Direction
- 27°
- Signal-to-Noise Ratio (SNR)
- 0 dB
- Signal-to-Interference plus Noise Ratio (SIR)
- -30 dB (Experiments 1 and 2)
- Candidate Count J
- 1500
- Number of Independent Trials
- 3~10 trials
3.2 Comparative Baseline Continuous MPDR (Capon Beamformer): Unconstrained Theoretical Upper Limit
Continuous Single-Mode MPDR (Tranter 2017, Lempel Gradient Descent): Theoretical upper limit of continuous phase single-mode
Classical Discrete Single-Mode MPDR + Local Search (K=1, b=1~6): Strongest classical baseline
Slepian K=2/K=3, b=1~6: Method proposed in this paper
3.3 Experimental Arrangement
Experiment 1: Depth of Null vs. Bit Number - Fixed SIR = -30 dB, to examine the impact of bit number on the directional null depth of each method. The deeper the null depth, the stronger the interference suppression.
Experiment 2: Beam Pattern Comparison - Plot the spatial response for each method within the range from -90° to 90°, with a focus on the zero-depth shape near the interference direction.
Experiment 3: Array Gain vs. Signal-to-Interference Ratio (SIR) - Fixed bit budget, to examine whether the array gain approaches an ideal value (approximately 15 dB) within the wide dynamic range of SIR from +60 dB to -60 dB.
IV. Results and Analysis
4.1 Zero-depth Variation with Bit Number Continuous MPDR can achieve a zero-depth of approximately -95 dB under -30 dB SIR conditions, with continuous single-mode MPDR around -55 dB - this represents the theoretical ceiling of single-mode constraints.
Classical discrete single-mode method (K=1) has a zero-depth of only about -10 dB at b=1, which improves gradually with increasing bit numbers, reaching close to the upper limit of continuous single-mode at b=6. The Slepian K=2 method can achieve a zero-depth of about -25 dB at b=1, and even better at K=3, reaching -35 dB - an improvement of about 15 dB and 25 dB respectively compared to the classical methods.
More noteworthy is that the zero-depth achieved by the Slepian K=3 at b=1 (with only two phase values) surpasses the performance of the classical K=1 at b=4 (with 16 phase values). This means that replacing a single-bit path with three-bit paths in three different directions can significantly reduce hardware complexity while maintaining or even improving performance.
4.2 Beam Pattern Analysis The beam pattern analysis reveals:
Continuous MPDR forms a sharp main lobe in the signal direction (5°), and in the interference direction (27°), it forms a deep null exceeding -90 dB, with sidelobes overall below -40 dB.
Classic single-mode b=1 method has a significant main lobe dispersion and shallow null directional attenuation (approximately -10 dB), significantly different from continuous MPDR.
The classic single-mode b=6 method shows noticeable improvement in null directional attenuation, but the sidelobe level is still higher than that of continuous MPDR.
The Slepian K=2 b=1 method has a deep null directional attenuation of about -25 dB in the interference direction, far superior to the classic b=1 method, with significantly better sidelobe control.
The Slepian K=3 b=1 method is already close to the beamforming pattern of continuous single-mode MPDR, confirming the quantitative advantage of the null directional attenuation depth indicator.
4.3 The robustness of array gain against signal-to-interference ratio (SIR) ranges from +60 dB to -60 dB:
Ideal array gain remains constant at 15 dB (only dependent on the number of elements M=32).
The classic single-mode b=2 method significantly drops in array gain when strong interference (SIR < -20 dB) occurs, severely limiting its performance.
The classic single-mode b=5 method shows significant improvement, but it is still slightly lower than the ideal value.
For the Slepian K=2/K=3 condition and under b=1, array gain closely approaches the ideal value of 15 dB across the entire SIR range, only slightly declining at the most extreme interference (SIR=-60 dB).
This result indicates that the Slepian multi-measurement vector MPDR not only can more effectively suppress fixed-direction interference but also has a greater adaptability to different levels of interference intensity compared to classical methods—a characteristic that is particularly important in real communication scenarios as the signal-to-interference ratio is often unknown and fluctuates sharply.
Discussion 5.1 The trade-off between K and b revealed through experimental data reveals an interesting engineering trade-off: increasing measurement channel count K offers a higher "performance gain/hardware cost ratio" over increasing single-channel precision b.
Taking the example of Slepian K=3 b=1 vs classical K=1 b=4, the former uses only 3 1-bit phase shifters (each requiring only one inverter), resulting in a very low total hardware cost; the latter requires a 4-bit phase shifter (necessitating a precision-grade cascading phase network). The former's null depth is actually superior to the latter. This fundamentally challenges the traditional belief that "beamforming performance is inevitably strongly correlated with phase shifter accuracy."
5.2 The Significance of Monotonicity Guarantee in Design_monotone Algorithms: The monotonicity guarantee is of great significance in practice for the design_monotone algorithm. In the low bit region (b=1,2,3), the naive greedy algorithm often encounters a paradox where variance is higher when b=2 than when b=1, making it difficult for engineers to determine "the optimal bit selection." The monotonicity guarantee fundamentally eliminates this issue, allowing any increase in bit count (no matter how small) to consistently bring about performance gains, simplifying system design and deployment decisions.
5.3 Limitations and Prospects The current research focuses on narrowband scenarios and single-target signals, which can be extended in three directions in the future:
1  
Broadband Expansion: Expanding the narrow-band Slepian MPDR to broadband OFDM systems, further reducing the equivalent measurement dimensions through frequency domain correlation.
2  
Multi-objective scenario: Multi-beam Shape Problem for Handling Multiple Signals Simultaneously
3  
Hardware Verification: Verify the algorithm's real-world performance on an FPGA or actual radio frequency frontend, especially considering the effects of phase shifters' non-idealities (phase errors, uneven insertion losses, etc.).
VI. Conclusion This paper systematically studies the low-precision phase-only controlled MPDR beamforming problem, and proposes a solution based on Slepian subspace dimension reduction and monotonicity guarantee recursive design. The experimental results show:
1  
K=2/K=3 bit measurement matrix Slepian MPDR respectively improves interference suppression depth by about 15 dB and 25 dB compared to classical methods.
2  
The Slepian scheme can achieve performance comparable to or even better than the classical b=4 scheme when b=1, providing a new path for ultra-low cost front-end design.
3  
The proposed design_monotone algorithm strictly ensures that performance improves monotonically with the number of bits, eliminating the risk of performance degradation in low bit regions.
4. In the wide dynamic signal-to-noise ratio range, the array gain of Slepian MPDR remains close to the theoretical ideal value.
This study provides a concrete success case for "compensating hardware precision deficiency with algorithms," offering practical guidance for the engineering deployment of large-scale arrays.
References: [1] DeLude et al., "Broadband Beamforming using Low-Bit Precision Dimensionality Reduction," IEEE JSTSP, 2025. [2] Tranter, Sidiropoulos, Fu, & Swami, "The Unimodular Waveform Design Problem," IEEE T-SP, 2017. [3] Capon, "High-Resolution Frequency-Wavenumber Spectrum Analysis," Proc. IEEE, 1969.

## Images

![Low-precision phase-only beamforming: when cheap hardware meets smart algorithms [with Python code]](cover.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
