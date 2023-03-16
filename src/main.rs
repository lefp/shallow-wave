use std::f32::consts::TAU;

use ocl::{enums::{MemObjectType, ImageChannelOrder, ImageChannelDataType}, MemFlags};
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
    image: ocl::Image<u8>,
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
                ocl_stuff.image.read(
                    // Reinterpret the &[u32] host buffer as &[u8] to read the CL image into it.
                    // This should be safe, because (on the common OSes+hardware) you can always expect to be
                    // able to fit four `u8`s in place of a `u32` without realigning, right?
                    unsafe { image_buffer.as_mut_slice().align_to_mut::<u8>().1 }
                ).enq().unwrap();
                graphics_context.set_buffer(image_buffer.as_ref(), WINDOW_WIDTH as u16, WINDOW_HEIGHT as u16);
            },
            _ => {},
        }
    })
}

fn set_up_opencl() -> OclStuff {
    let src = r#"
        // t is just a time paramter to get a changing image
        kernel void render(float t, write_only image2d_t image) {
            uint x = get_global_id(0);
            uint y = get_global_id(1);
            uint width  = get_image_width (image);
            uint height = get_image_height(image);

            // `some_norm` is in [0,1]
            float some_norm = (float)(x*x + y*y)
                            / (float)(width*width + height*height);

            float4 color = 0.f;
            color.r = some_norm * fabs(sin(t));
            color.g = (1.f - color.r) * (1.f - some_norm) * fabs(cos(t));
            color.b = 1.f - (color.r + color.g);
            write_imagef(image, (int2)(x, y), color);
        }
    "#;

    let pro_que = ocl::ProQue::builder()
                  .src(src)
                  .dims((WINDOW_WIDTH, WINDOW_HEIGHT))
                  .build().unwrap();

    let image = ocl::Image::<u8>::builder()
                .queue(pro_que.queue().clone())
                .image_type(MemObjectType::Image2d)
                // The `softbuffer` crate expects each pixel to be `0rgb`, where each component is 8 bits.
                // Note that the first component is expected to be 0!
                /* @todo looks like Argb is backwards and Bgra works as expected. I think the `align_to_mut`
                cast causes the bytes to be backwards because intel is little-endian */
                .channel_order(ImageChannelOrder::Argb)
                // UnormInt8: each component is a u8 in 0-255, but the 'norm' part means that our kernel code
                // should treat it as a float in 0-1. OpenCL handles the conversion automatically.
                .channel_data_type(ImageChannelDataType::UnormInt8)
                .dims((WINDOW_WIDTH, WINDOW_HEIGHT))
                .flags(
                    // note: CL_MEM_KERNEL_READ_AND_WRITE support is not guaranteed under OpenCL 3.0
                    MemFlags::WRITE_ONLY | // kernel will only render to it
                    MemFlags::HOST_READ_ONLY // host will read so that `softbuffer` can display the image
                )
                .build().expect("Failed to build OpenCL image.");

    let render_kernel = pro_que.kernel_builder("render")
                        .arg_named("t", 0f32)
                        .arg(&image)
                        .build().unwrap();

    OclStuff { pro_que, image, render_kernel }
}

