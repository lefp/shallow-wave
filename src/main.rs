const WINDOW_WIDTH:  usize = 800;
const WINDOW_HEIGHT: usize = 600;
const SPACIAL_DOMAIN_SIZE: f32 = 100.;
const N_GRIDPOINTS: usize = 1000;
const TIME_STEP: f32 = 0.001;
const GAUSSIAN_INITIALIZER_DECAY: f32 = 0.005;
const AXIS_BOUNDS: [f32; 2] = [0., 1.]; // bottom and top of the wave height axis to be displayed
const BACKGROUND_FLOW_SPEED: f32 = 1.;
const RENDER_FPS: f32 = 60.;

// derived constants
const SPACIAL_STEP: f32 = SPACIAL_DOMAIN_SIZE / N_GRIDPOINTS as f32;
const GRID_ENDPOINT: f32 = SPACIAL_STEP * (N_GRIDPOINTS - 1) as f32;
const GRID_CENTERPOINT: f32 = 0.5 * GRID_ENDPOINT;
const RENDER_INTERVAL: f32 = 1. / RENDER_FPS;

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
        WindowEvent,
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
        let initial_h_values: Vec<f32> = (0..N_GRIDPOINTS).map(|i| {
            f32::exp(-GAUSSIAN_INITIALIZER_DECAY * (i as f32 * SPACIAL_STEP - GRID_CENTERPOINT).powi(2))
        }).collect();
        set_up_opencl(&initial_h_values, AXIS_BOUNDS)
    };

    // each u32 represents a color: 8 bits of nothing, then 8 bits each of R, G, B
    let mut image_hostbuffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut iter = 0;
    let mut iter_display_timer = time::Instant::now();
    let mut frame_timer = time::Instant::now();
    let frame_duration = Duration::from_secs_f32(RENDER_INTERVAL);
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll(); // Continuously runs the event loop, as opposed to `set_wait`

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { control_flow.set_exit(); },
            Event::MainEventsCleared => { // APPLICATION UPDATE CODE GOES HERE
                unsafe { ocl_stuff.iteration_kernel.cmd().enq().unwrap(); }
                // std::thread::sleep(std::time::Duration::from_millis(10));

                // print iteration number if needed
                iter += 1;
                if iter_display_timer.elapsed() >= Duration::from_secs(1) {
                    println!("iter: {iter}");
                    iter_display_timer = time::Instant::now();
                }

                // update display if needed. We don't do this on every iteration because it's slow
                if frame_timer.elapsed() >= frame_duration { window.request_redraw(); }
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

fn set_up_opencl(initial_h_values: &[f32], axis_bounds: [f32; 2]) -> OclStuff {
    let src = {
        let mut src = String::new();
        File::open("src/shallow_wave.cl").unwrap().read_to_string(&mut src).unwrap();
        src
    };

    let pro_que = ocl::ProQue::builder()
        .src(src)
        .dims((WINDOW_WIDTH, WINDOW_HEIGHT))
        .build().unwrap();

    let h_buffer = ocl::Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(initial_h_values.len())
        .copy_host_slice(initial_h_values)
        .flags(MemFlags::HOST_NO_ACCESS | MemFlags::READ_WRITE)
        .build().expect("Failed to build h buffer.");

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
        .global_work_size(initial_h_values.len())
        .arg_named("h", h_buffer.clone())
        .arg_named("dx", SPACIAL_STEP)
        .arg_named("dt", TIME_STEP)
        .arg_named("background_flow_speed", BACKGROUND_FLOW_SPEED)
        .build().expect("Failed to build iteration kernel.");

    let render_kernel = pro_que.kernel_builder("render")
        .global_work_size((WINDOW_WIDTH, WINDOW_HEIGHT))
        .arg_named("render_target", image.clone())
        .arg_named("h", h_buffer.clone())
        .arg_named("h_size", initial_h_values.len() as u32)
        .arg_named("axis_min", axis_bounds[0])
        .arg_named("axis_max", axis_bounds[1])
        .build().expect("Failed to build render kernel.");

    OclStuff { pro_que, image, iteration_kernel, render_kernel }
}

