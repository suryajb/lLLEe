# lLLEe
learn LLE easily

Motivation: this code is optimized for LLE (Lugiato-Lefever Equation) simulation and is 3x faster (could be more) than the only open source LLE code out there I could find ([pyLLE](https://github.com/gregmoille/pyLLE) developed by NIST scientists) which is already highly optimized for simulation of Kerr microring soliton dynamics (lLLEe can be faster depending on computer and simulation parameters). Additionally if you're not that familiar with coding and the math behind LLE, lLLEe might be easier to follow since the main computation is like a few lines written in python. If you're already convinced, you can skip to the "How to use" section.

Short and possibly inaccurate comparison with pyLLE: 
If you just want to do a quick simulation of a comb spectrum with known parameters, you should look into pyLLE which currently probably has an easier interface.
If you want to learn about the inner-workings of the LLE, or wish to customize it to your own needs, and would like to simulate a comb spectrum extremely efficiently (on the level of C code and parallelized) due to some physical parameters of your system (high input power, large number of modes, small time steps/long simulation time...), but you don't have extensive coding experience, you might want to look into lLLEe.

## Why did I do this?

lLLEe is a simulation software for the driven, damped, detuned, nonlinear Schrodinger equation, AKA [Lugiato-Lefever equation](https://en.wikipedia.org/wiki/Lugiato%E2%80%93Lefever_equation).  Its use is widespread in the nonlinear optics comunity and getting a good hang of it is important for any beginner in the space.  This code is designed to optimize the numerical simulation of the LLE while being very easy to customize and read for people of varying levels of experience in python.  This is because for different applications and material platforms users may discover the need to add additional physical effects (thermal, Raman, photorefractive, second order nonlinear processes... etc.), and being able to do so with ease is quite important for the scientific community.  Therefore most of the code written here is in python, and all the "new" physics one may wish to add into the LLE can be done with relative ease (inside a concentrated section of the code).

In order for this code to be readable for people of varying levels of experience in python and comfort with the details of the LLE, the computation has been implemented in 4 ways with increasing levels of complexity.  The more complex the code, the more optimized the simulation.  This was done with the intention of allowing a user to first experiment with the easier code before slowly optimizing it (this is also what I did to systematically get to the final most optimized version).  Even the most complex code is not that hard to read because it should still be readable to anyone who is familiar with python.

The following are 4 ways the simulation was coded in:
1. Unoptimized - this version uses numpy so it wouldn't be so slow.  It's basically the equation written out in code and integrated in time.  Even the symbols should look familiar if the user has first read some of the earlier papers<sup>[1](https://doi.org/10.1103/PhysRevA.82.033801)</sup> on the split step algorithm used in solving the LLE and soliton/Kerr comb simulations.  
2. FFT optimized - this version uses pyFFTW, which runs the fast fourier transforms (FFT) in optimized C code, and is ~1.6x faster than (1.).
3. Numba optimized - this version, on top of being FFT optimized, uses Numba to do some of the heavy lifting (converts computationally difficult code into optimized machine language, which is compiled at run-time).  This is ~2.2x faster than (1.)
4. Cython optmized - this is the most optimized code which on top of doing what Numba essentially does (defines all types in advance, lays out the ground work for the loops in C code), unwraps some of the numpy computations (np.exp) and allows a lot of the calculations to be done in parallel.  Users should figure it out for themselves how many threads to use. Using 8 threads on my own machine, I was able to achieve >4x improvement over the unoptimized version (1.).  Comparing this version with the only open source LLE simulation software out there that I could find (with my 6core/12threads CPU), which is already advertised to be >4x faster than a pure python implementation, Cython-lLLEe was an additional 3x faster than pyLLE developed by scientists at NIST. PyLLE is written using a julia backend to optimize computations, which is already approximately 4 times faster than an only python approach, I believe it also uses a semi-adaptive time step to improve accuracy of the simulation which is something lLLEe lacks.

If only the most basic version of the LLE needs to be solved, then you only need to run Cython-lLLEe.

## How to use
coming soon...
