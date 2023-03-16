use std::f32::consts::TAU;

use winit::{
    event_loop::{
        EventLoop,
        ControlFlow,
    },
    window::{Window, WindowBuilder},
    event::{
        Event,
        WindowEvent,
    },
    dpi::LogicalSize,
};
use softbuffer::GraphicsContext;

const WINDOW_WIDTH:  usize = 800;
const WINDOW_HEIGHT: usize = 600;

struct OclStuff {
    pro_que: ocl::ProQue,
    image_buffer: ocl::Buffer<u32>,
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
    let ocl_stuff = set_up_opencl();
    // each u32 represents a color: 8 bits of nothing, then 8 bits each of R, G, B
    let mut image_buffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];
    let mut t = 0f32; // time parameter

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll(); // Continuously runs the event loop, as opposed to `set_wait`

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { control_flow.set_exit(); },
            Event::MainEventsCleared => { // APPLICATION UPDATE CODE GOES HERE
                /* @todo we redraw on every iteration of the event loop for now. If that changes, move the
                drawing code to the RedrawRequested arm of the match statement.
                */

                // update state
                t = (t + 0.001) % TAU;

                // render and display image
                ocl_stuff.render_kernel.set_arg("t", t).unwrap();
                unsafe { ocl_stuff.render_kernel.enq().unwrap(); }
                ocl_stuff.image_buffer.read(&mut image_buffer).enq().unwrap();
                graphics_context.set_buffer(image_buffer.as_ref(), WINDOW_WIDTH as u16, WINDOW_HEIGHT as u16);
            },
            _ => {},
        }
    })
}

fn set_up_opencl() -> OclStuff {
    let src = r#"
        // t is just a time paramter to get a changing image
        kernel void render(float t, global uint* buffer, uint image_width, uint image_height) {
            uint linear_index = get_global_id(0);
            uint x = linear_index % image_width;
            uint y = linear_index / image_width;

            // `some_norm` is in [0,1]
            float some_norm = (float)(x*x + y*y)
                            / (float)(image_width*image_width + image_height*image_height);

            uint red   = (uint)(255.f * some_norm * fabs(sin(t)));
            uint green = (uint)( (float)(255 - red) * ((1.f - some_norm) * fabs(cos(t))) );
            uint blue = 255 - (red + green);
            buffer[linear_index] = (red << 16) | (green << 8) | blue;
        }
    "#;

    let pro_que = ocl::ProQue::builder()
                  .src(src)
                  .dims(WINDOW_WIDTH * WINDOW_HEIGHT)
                  .build().unwrap();

    let image_buffer = pro_que.create_buffer::<u32>().unwrap();

    let render_kernel = pro_que.kernel_builder("render")
                        .arg_named("t", 0f32)
                        .arg(&image_buffer)
                        .arg(WINDOW_WIDTH  as u32)
                        .arg(WINDOW_HEIGHT as u32)
                        .build().unwrap();

    OclStuff { pro_que, image_buffer, render_kernel }
}

