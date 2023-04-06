# shallow-wave
Shallow wave simulation using Rust+OpenCL.

Rendering is also done using OpenCL, an idea taken from [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) that really simplifies the graphics code.

Next steps:
* 2 horizontal dimensions
* Viscosity
* A different fluid model. The shallow-water equations cause the wavefronts to get steeper over time, but the equations also break down when the wavefronts become very steep.

![1d_wave](https://user-images.githubusercontent.com/70862148/230284788-efc1cc72-0814-47f7-a1d6-bb2e362215a4.gif)