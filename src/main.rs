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

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
                .with_resizable(false)
                .with_inner_size(LogicalSize::new(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32))
                .build(&event_loop)
                .expect("Failed to build window.");
    // the graphics context is the abstraction through which images are written to the window, I think
    let mut graphics_context = unsafe { GraphicsContext::new(&window, &window) }.unwrap();

    // each u32 represents a color: 8 bits of nothing, then 8 bits each of R, G, B
    let mut image_buffer = Box::new([0u32; WINDOW_WIDTH * WINDOW_HEIGHT]);

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll(); // Continuously runs the event loop, as opposed to `set_wait`

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { control_flow.set_exit(); },
            Event::MainEventsCleared => { // APPLICATION UPDATE CODE GOES HERE
                /* @todo we redraw on every iteration of the event loop for now. If that changes, move the
                drawing code to the RedrawRequested arm of the match statement.
                */

                for linear_index in 0..image_buffer.len() {
                    let x = linear_index % WINDOW_WIDTH;
                    let y = linear_index / WINDOW_WIDTH;

                    // `some_norm` is in [0,1]
                    let some_norm = (x*x + y*y) as f32
                                    / (WINDOW_WIDTH.pow(2) + WINDOW_HEIGHT.pow(2)) as f32;

                    let red   = 0u32;
                    let green = (255. * some_norm) as u32;
                    let blue  = 255 - green;

                    image_buffer[linear_index] = (red << 16) | (green << 8) | blue;
                }

                graphics_context.set_buffer(image_buffer.as_ref(), WINDOW_WIDTH as u16, WINDOW_HEIGHT as u16);
            },
            _ => {},
        }
    })
}
