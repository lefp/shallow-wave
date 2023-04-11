const WINDOW_WIDTH:  usize = 800;
const SPATIAL_DOMAIN_SIZE_X: f32 = 25.;
const SPATIAL_DOMAIN_SIZE_Y: f32 = 25.;
const N_GRIDPOINTS_X: usize = 100;
const N_GRIDPOINTS_Y: usize = 100;
const TIME_STEP: f32 = 0.00001;
const RENDER_FPS: f32 = 60.;
const INITIALLY_PAUSED: bool = true; // whether application wait for an unpause before running simulation

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
    time::{self, Duration},
    sync::mpsc::{self, TryRecvError, TrySendError},
    thread,
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
    image: ocl::Image<u32>,
    simulation_kernel: ocl::Kernel,
    render_kernel:     ocl::Kernel,
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

    let OclStuff { simulation_kernel, render_kernel, image } = {
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

    // Display the initial state, so that the window isn't blank if the program is paused on startup.
    unsafe { render_kernel.enq().unwrap(); }
    image.read(&mut image_hostbuffer)
        .queue(render_kernel.default_queue().unwrap()) // use same queue, so read waits for render to complete
        .enq().unwrap();
    graphics_context.set_buffer(&image_hostbuffer, WINDOW_WIDTH as u16, WINDOW_HEIGHT as u16);

    // set up communication channels
    let (pause_toggle_tx, pause_toggle_rx) = mpsc::sync_channel::<()>(0);
    let (render_request_tx, render_request_rx) = mpsc::sync_channel::<()>(1);
    /*
    channel buffer size 1 so that the simulation thread can non-blockingly send back the render status event
    and continue feeding simulation commands to the hungry GPU
    */
    let (render_status_tx, render_status_rx) = mpsc::sync_channel::<ocl::EventList>(1);

    // set up simulation thread
    let _sim_and_render_thread = thread::spawn(move || {
        let mut iter = 0;
        let mut last_printed_iter = 0;
        let mut iter_print_timer = time::Instant::now();

        if INITIALLY_PAUSED { pause_toggle_rx.recv().unwrap(); } // wait for an unpause before starting
        loop {
            // check if we were told to pause
            match pause_toggle_rx.try_recv() {
                Err(TryRecvError::Disconnected) => { panic!() },
                Err(TryRecvError::Empty) => {}, // no message, continue as normal
                Ok(()) => { pause_toggle_rx.recv().unwrap(); }, // pause until we receive another toggle
            }
            // check if we were told to render
            match render_request_rx.try_recv() {
                Err(TryRecvError::Disconnected) => { panic!() },
                Err(TryRecvError::Empty) => {}, // no message, continue as normal
                Ok(()) => {
                    let mut render_status = ocl::EventList::with_capacity(1); // @todo is repeatedly creating a new EventList inefficient?
                    unsafe { render_kernel.cmd().enew(&mut render_status).enq().unwrap(); }
                    match render_status_tx.try_send(render_status) {
                        Err(TrySendError::Disconnected(_)) => { panic!() },
                        // we really don't want the receiver to slow down the simulation by blocking us
                        Err(TrySendError::Full(_)) => { panic!("Blocked while sending render status."); },
                        Ok(()) => {},
                    }
                },
            }

            // continue running the simulation
            unsafe { simulation_kernel.enq().unwrap(); }

            // print iteration number if needed
            iter += 1;
            if iter_print_timer.elapsed() >= Duration::from_secs(1) {
                println!("iter: {iter:10} | iters per second: {}", iter - last_printed_iter);
                last_printed_iter = iter;
                iter_print_timer = time::Instant::now();
            }
        }
    });

    let mut paused: bool = INITIALLY_PAUSED;
    let mut frame_timer = time::Instant::now();
    let frame_duration = Duration::from_secs_f32(RENDER_INTERVAL);
    // note: `control_flow` defaults to `ControlFlow::Poll` before the event loop's first iteration
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                // @todo cleanly stop other threads here?
                control_flow.set_exit();
            },
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if let KeyboardInput {
                    virtual_keycode: Some(VirtualKeyCode::P),
                    state: ElementState::Pressed,
                    ..
                } = input {
                    paused = !paused;
                    pause_toggle_tx.send(()).unwrap();
                    if paused { control_flow.set_wait(); } // don't waste CPU while paused
                    else { control_flow.set_poll(); }
                };
            },
            Event::MainEventsCleared => { // APPLICATION UPDATE CODE GOES HERE
                /* Update display if needed.
                We do this at a chosen framerate instead of on every loop iteration, because it's slow.
                */
                if !paused && frame_timer.elapsed() >= frame_duration {
                    /* We do the render+read here instead of in RedrawRequested; this way, external redraw
                    requests don't affect the render+read rate, WE decide when to do it, giving us better
                    control over performance.
                    Plus, why render more often than the chosen framerate? If there's a redraw request between
                    frames, just show the old frame again. This isn't a high-end videogame, we don't care
                    about keeping up high-refresh-rate monitors.
                    */
                    /* @todo we should send the previous read event in the render request, so that the
                    renderer can wait for it... but we really don't want the simulator thread to wait on a
                    read, so we should just not request a re-render if we still have a read in progress. In
                    fact, this problem never occurs because because the read is blocking, so it's impossible
                    to send a re-render request until the read is finished. We could still send the read event
                    with the re-render request in case we accidentally break that invariant, and just have the
                    sim kernel panic if the read state isn't CL_COMPLETE. */
                    render_request_tx.send(()).unwrap();
                    let render_completed = render_status_rx.recv().unwrap();
                    image.read(image_hostbuffer.as_mut_slice()).ewait(&render_completed).enq().unwrap();
                    window.request_redraw();
                    frame_timer = time::Instant::now();
                }
            },
            Event::RedrawRequested(..) => {
                graphics_context.set_buffer(image_hostbuffer.as_ref(), WINDOW_WIDTH as u16, WINDOW_HEIGHT as u16);
            },
            _ => {},
        }
    });
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

    // @todo better platform and device selection logic
    let platform = ocl::Platform::first().unwrap();
    let device = ocl::Device::first(&platform).unwrap();
    println!("Chose device '{}'", device.name().unwrap());

    let context = ocl::Context::builder()
        .devices(&device)
        .build().unwrap();

    let program = ocl::Program::builder()
        .src_file("src/shallow_wave.cl")
        .cmplr_def_f32("TIME_STEP", TIME_STEP)
        .cmplr_def_f32("SPATIAL_STEP_X", SPATIAL_STEP_X)
        .cmplr_def_f32("SPATIAL_STEP_Y", SPATIAL_STEP_Y)
        .cmplr_def("N_GRIDPOINTS_X", N_GRIDPOINTS_X as i32)
        .cmplr_def("N_GRIDPOINTS_Y", N_GRIDPOINTS_Y as i32)
        .build(&context).expect("Failed to build OpenCL program.");

    /* @todo is there an upper limit on how many commands we can enqueue?
    Do they just take up more and more memory?
    Does the enqueueing function block when the queue is full? Would that cause issues like lag?
    The OpenCL 3.0 spec doesn't seem to say anything about this.
    */
    let sim_and_render_queue = ocl::Queue::new(&context, device.clone(), None).unwrap();
    let data_transfer_queue  = ocl::Queue::new(&context, device.clone(), None).unwrap();

    let h_buffer = ocl::Buffer::<f32>::builder()
        .queue(data_transfer_queue.clone())
        .len(TOTAL_N_GRIDPOINTS)
        .copy_host_slice(initial_h_values)
        .flags(MemFlags::HOST_NO_ACCESS | MemFlags::READ_WRITE)
        .build().expect("Failed to build h buffer.");

    let w_buffer = ocl::Buffer::<ocl::prm::Float2>::builder()
        .queue(data_transfer_queue.clone())
        .len(TOTAL_N_GRIDPOINTS)
        .fill_val(ocl::prm::Float2::zero()) // 0-initialize
        .flags(MemFlags::HOST_NO_ACCESS | MemFlags::READ_WRITE)
        .build().expect("Failed to build w buffer.");

    let image = ocl::Image::<u32>::builder()
        .queue(data_transfer_queue.clone())
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

    let simulation_kernel = ocl::Kernel::builder()
        .program(&program)
        .name("iterate")
        .queue(sim_and_render_queue.clone())
        .global_work_size([N_GRIDPOINTS_X, N_GRIDPOINTS_Y])
        .arg_named("h", h_buffer.clone())
        .arg_named("w", w_buffer.clone())
        .build().expect("Failed to build iteration kernel.");

    let render_kernel = ocl::Kernel::builder()
        .program(&program)
        .name("render")
        .queue(sim_and_render_queue.clone())
        .global_work_size((WINDOW_WIDTH, WINDOW_HEIGHT))
        .arg_named("render_target", image.clone())
        .arg_named("h", h_buffer.clone())
        .arg_named("axis_min", axis_bounds[0])
        .arg_named("axis_max", axis_bounds[1])
        .build().expect("Failed to build render kernel.");

    OclStuff { image, simulation_kernel, render_kernel }
}

