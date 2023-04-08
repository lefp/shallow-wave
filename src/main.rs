const WINDOW_WIDTH:  usize = 800;
const SPATIAL_DOMAIN_SIZE_X: f32 = 25.;
const SPATIAL_DOMAIN_SIZE_Y: f32 = 25.;
const N_GRIDPOINTS_X: usize = 100;
const N_GRIDPOINTS_Y: usize = 100;
const TIME_STEP: f32 = 0.00001;
const RENDER_FPS: f32 = 60.;

// initial conditions
const FLUID_DEPTH: f32 = 1.0; // do not set to 0, else expect freaky behavior
const INIT_WAVE_HEIGHT: f32 = 0.1; // height of the wave above the rest of the fluid surface
const INIT_STDDEV_X: f32 = 1.0;
const INIT_STDDEV_Y: f32 = 1.0;
const INIT_WAVE_CENTERPOINT_RELATIVE_X: f32 = 0.75; // "relative" meaning "in normalized [0, 1] coordinates"
const INIT_WAVE_CENTERPOINT_RELATIVE_Y: f32 = 0.50;

// derived constants
const TOTAL_N_GRIDPOINTS: usize = N_GRIDPOINTS_X * N_GRIDPOINTS_Y;
const SPATIAL_STEP_X: f32 = SPATIAL_DOMAIN_SIZE_X / N_GRIDPOINTS_X as f32;
const SPATIAL_STEP_Y: f32 = SPATIAL_DOMAIN_SIZE_Y / N_GRIDPOINTS_Y as f32;
const INIT_WAVE_CENTERPOINT_X: f32 = INIT_WAVE_CENTERPOINT_RELATIVE_X * SPATIAL_DOMAIN_SIZE_X;
const INIT_WAVE_CENTERPOINT_Y: f32 = INIT_WAVE_CENTERPOINT_RELATIVE_Y * SPATIAL_DOMAIN_SIZE_Y;
const RENDER_INTERVAL: f32 = 1. / RENDER_FPS;
const WINDOW_HEIGHT: usize = ((SPATIAL_DOMAIN_SIZE_Y / SPATIAL_DOMAIN_SIZE_X) * WINDOW_WIDTH as f32) as usize;

use std::{
    fs::File,
    io::Read,
    time::{self, Duration},
};

use ocl::{
    enums::{MemObjectType, ImageChannelOrder, ImageChannelDataType},
    MemFlags,
};
use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
    event::{
        Event,
        WindowEvent, KeyboardInput, ElementState, VirtualKeyCode,
    },
    dpi::LogicalSize,
};
use softbuffer::GraphicsContext;

struct OclStuff {
    pro_que: ocl::ProQue,
    image: ocl::Image<u32>,
    iteration_kernel: ocl::Kernel,
    render_kernel: ocl::Kernel,
}

fn gaussian2d<F: num::traits::Float>(xcoord: F, ycoord: F, stddev_x: F, stddev_y: F, center_x: F, center_y: F) -> F {
    F::exp( -F::from(0.5).unwrap() * (
        ((xcoord - center_x) / stddev_x).powi(2) +
        ((ycoord - center_y) / stddev_y).powi(2)
    ))
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_resizable(false)
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32))
        .build(&event_loop)
        .expect("Failed to build window.");
    // the graphics context is the abstraction through which images are written to the window, I think
    let mut graphics_context = unsafe { GraphicsContext::new(&window, &window) }.unwrap();

    let ocl_stuff = {
        let initial_h_values: Vec<f32> = (0..TOTAL_N_GRIDPOINTS).map(|i| {
            let xcoord = (i % N_GRIDPOINTS_X) as f32 * SPATIAL_STEP_X;
            let ycoord = (i / N_GRIDPOINTS_X) as f32 * SPATIAL_STEP_Y;
            FLUID_DEPTH +
            INIT_WAVE_HEIGHT *
            gaussian2d(
                xcoord, ycoord,
                INIT_STDDEV_X, INIT_STDDEV_Y,
                INIT_WAVE_CENTERPOINT_X, INIT_WAVE_CENTERPOINT_Y
            )
        }).collect();
        let min_initial_h = initial_h_values.iter().copied().reduce(f32::min).unwrap();
        let max_initial_h = initial_h_values.iter().copied().reduce(f32::max).unwrap();
        set_up_opencl(&initial_h_values, [min_initial_h - 0.1*(max_initial_h - min_initial_h), max_initial_h])
    };

    // each u32 represents a color: 8 bits of nothing, then 8 bits each of R, G, B
    let mut image_hostbuffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut paused = true;
    let mut iter = 0;
    let mut iter_display_timer = time::Instant::now();
    let mut frame_timer = time::Instant::now();
    let frame_duration = Duration::from_secs_f32(RENDER_INTERVAL);
    // note: `control_flow` defaults to `ControlFlow::Poll` before the event loop's first iteration
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { control_flow.set_exit(); },
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if let KeyboardInput {
                    virtual_keycode: Some(VirtualKeyCode::P),
                    state: ElementState::Pressed,
                    ..
                } = input {
                    paused = !paused;
                    if paused { control_flow.set_wait(); } // don't waste CPU while paused
                    else { control_flow.set_poll(); }
                };
            },
            Event::MainEventsCleared => { // APPLICATION UPDATE CODE GOES HERE
                if !paused {
                    unsafe { ocl_stuff.iteration_kernel.enq().unwrap(); }

                    // print iteration number if needed
                    iter += 1;
                    if iter_display_timer.elapsed() >= Duration::from_secs(1) {
                        println!("iter: {iter}");
                        iter_display_timer = time::Instant::now();
                    }

                    // update display if needed. We don't do this on every iteration because it's slow
                    if frame_timer.elapsed() >= frame_duration { window.request_redraw(); }
                    // std::thread::sleep(Duration::from_secs_f32(0.01f32)); // @debug
                }
            },
            Event::RedrawRequested(..) => {
                unsafe { ocl_stuff.render_kernel.enq().unwrap(); }
                ocl_stuff.image.read(image_hostbuffer.as_mut_slice()).enq().unwrap();
                graphics_context.set_buffer(image_hostbuffer.as_ref(), WINDOW_WIDTH as u16, WINDOW_HEIGHT as u16);
                frame_timer = time::Instant::now();
            },
            _ => {},
        }
    })
}

/* A stupid hack for defining float constants in kernels at kernel-compile-time.
A little information is probably lost in the `float -> decimal string -> float` conversion.
I don't know of a better way to do this.
@todo One option is SPIR-V specialization constants. But that isn't supported in OpenCL 1.2 and the `ocl`
crate, so you'll have to use a different crate and explicitly query support from the OpenCL runtime/device.
*/
trait OclCompileTimeF32ConstantHack {
    fn cmplr_def_f32<'a, S: Into<String>>(&'a mut self, name: S, val: f32) -> &'a mut Self;
}
impl<'b> OclCompileTimeF32ConstantHack for ocl::builders::ProgramBuilder<'b> {
    fn cmplr_def_f32<'a, S: Into<String>>(&'a mut self, name: S, val: f32) -> &'a mut Self {
        // exponential notation is to avoid losing information for numbers very close 0
        let opt = format!("-D{}={:.100e}", name.into(), val);
        println!("{opt}"); // @debug
        self.cmplr_opt(opt)
    }
}

fn set_up_opencl(initial_h_values: &[f32], axis_bounds: [f32; 2]) -> OclStuff {
    /* @note: you MUST ensure that the OpenCL and kernel argument types defined here are the same as those
    defined in the OpenCL program/kernel source code.
    @todo mark this function unsafe, as the Rust-to-OpenCL-C interface is not type-safe.
    */

    assert_eq!(initial_h_values.len(), TOTAL_N_GRIDPOINTS);

    let mut prog_builder = ocl::Program::builder();
    prog_builder
        .src_file("src/shallow_wave.cl")
        .cmplr_def_f32("TIME_STEP", TIME_STEP)
        .cmplr_def_f32("SPATIAL_STEP_X", SPATIAL_STEP_X)
        .cmplr_def_f32("SPATIAL_STEP_Y", SPATIAL_STEP_Y)
        .cmplr_def("N_GRIDPOINTS_X", N_GRIDPOINTS_X as i32)
        .cmplr_def("N_GRIDPOINTS_Y", N_GRIDPOINTS_Y as i32);
    let pro_que = ocl::ProQue::builder()
        .prog_bldr(prog_builder)
        .build().unwrap();

    println!("{}", pro_que.device().name().unwrap()); // @debug

    let h_buffer = ocl::Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(TOTAL_N_GRIDPOINTS)
        .copy_host_slice(initial_h_values)
        .flags(MemFlags::HOST_NO_ACCESS | MemFlags::READ_WRITE)
        .build().expect("Failed to build h buffer.");

    let w_buffer = ocl::Buffer::<ocl::prm::Float2>::builder()
        .queue(pro_que.queue().clone())
        .len(TOTAL_N_GRIDPOINTS)
        .fill_val(ocl::prm::Float2::zero()) // 0-initialize
        .flags(MemFlags::HOST_NO_ACCESS | MemFlags::READ_WRITE)
        .build().expect("Failed to build w buffer.");

    let image = ocl::Image::<u32>::builder()
        .queue(pro_que.queue().clone())
        .image_type(MemObjectType::Image2d)
        // `softbuffer` expects each pixel to be a u32 `0rgb`, where each component is 8 bits.
        // Note that the first component is expected to be 0!
        /* Using single-channel UnsignedInt32 for the OpenCL image because:
            The alternative is using 4-channel 8-bit, and reinterpreting the &[u8] as &[u32] after
            copying the data to host. The problem is that the correctness of the reinterpretation
            depends on the endianness of the CPU architecture, which is not something I want to keep
            track of.
        */
        .channel_order(ImageChannelOrder::R)
        .channel_data_type(ImageChannelDataType::UnsignedInt32)
        .dims((WINDOW_WIDTH, WINDOW_HEIGHT))
        .flags(
            // note: CL_MEM_KERNEL_READ_AND_WRITE support is not guaranteed under OpenCL 3.0
            MemFlags::WRITE_ONLY | // kernel will only render to it
            MemFlags::HOST_READ_ONLY // host will read so that `softbuffer` can display the image
        )
        .build().expect("Failed to build OpenCL image.");

    let iteration_kernel = pro_que.kernel_builder("iterate")
        .global_work_size([N_GRIDPOINTS_X, N_GRIDPOINTS_Y])
        .arg_named("h", h_buffer.clone())
        .arg_named("w", w_buffer.clone())
        .build().expect("Failed to build iteration kernel.");

    let render_kernel = pro_que.kernel_builder("render")
        .global_work_size((WINDOW_WIDTH, WINDOW_HEIGHT))
        .arg_named("render_target", image.clone())
        .arg_named("h", h_buffer.clone())
        .arg_named("axis_min", axis_bounds[0])
        .arg_named("axis_max", axis_bounds[1])
        .build().expect("Failed to build render kernel.");

    OclStuff { pro_que, image, iteration_kernel, render_kernel }
}

