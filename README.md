# shallow-wave
Shallow water simulation using Rust+OpenCL.

Rendering is also done using OpenCL, an idea taken from [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) that really simplifies the graphics code.

The main goal is to develop skills toward GPU cluster computing, and (with lower priority) numerical methods. I'm also hoping to prove to myself that there is a decent alternative to Fortran/C/C++ for HPC.

Next steps:
* Multi-GPU support
* Numerical methods requiring more inter-invocation and inter-GPU communication
* A different fluid model. The shallow-water equations cause the wavefronts to get steeper over time, but the equations also break down when the wavefronts become very steep. Severely limiting the time-domain makes the visualizations boring.

![1d_wave](https://user-images.githubusercontent.com/70862148/230284788-efc1cc72-0814-47f7-a1d6-bb2e362215a4.gif)

![2d_wave](https://user-images.githubusercontent.com/70862148/230698084-a7714001-a8d8-4def-bb03-62e7388082cf.gif)
