/* @todo dt and dx should be specialization constants; @todo but those are only available in OpenCL 2.2+, and
    I don't think the `ocl` crate even supports them. Might have to do some cursed C code creation at runtime,
    or use the -D flag in the kernel/program compilation step
*/
kernel void iterate(global float* h, float dx, float dt, float background_flow_speed) {
    uint gid = get_global_id(0);
    uint grid_size = get_global_size(0);

    // compute dh using centered difference
    float dh;
    if (gid == 0) dh = (h[1] - h[grid_size - 1]) * 0.5;
    else if (gid == grid_size - 1) dh = (h[0] - h[grid_size - 2]) * 0.5;
    else dh = (h[gid + 1] - h[gid - 1]) * 0.5;

    barrier(CLK_GLOBAL_MEM_FENCE); // make sure all invocations have read from `h` before writing back to it
    h[gid] -= background_flow_speed * (dh / dx) * dt;
}

// RENDERING CODE --------------------------------------------------------------------------------------------

// Important: expects input to be in [0,1]. Otherwise expect nonsensical results.
uint normalized_float3_to_u32(float3 color) {
    color *= 255.f;
    // @note indexing using `.r` etc is supposedly an OpenCL 3.0 feature
    return ((uint)color.r << 16) | ((uint)color.g << 8) | (uint)color.b;
}

// Converts the float3 `color` to a u32 and writes the result to `image` at `coord`.
void convert_and_write_image(write_only image2d_t image, int2 coord, float3 color) {
    write_imageui(image, coord, normalized_float3_to_u32(color));
};

/* Renders `h` to `render_target`, displaying between `axis_min` and `axis_max`.
`h_size` is the length of the `h` array.
Expects `render_target` to be a single-channel `UnsignedInt32` of the form `00000000rrrrrrrrggggggggbbbbbbbb`
*/
// @todo maybe some of these should be specialization constants?
kernel void render(write_only image2d_t render_target, global float* h, uint h_size, float axis_min, float axis_max) {
    int pixel_xcoord = get_global_id(0);
    int pixel_ycoord = get_global_id(1);

    int width  = get_image_width(render_target);
    int height = get_image_height(render_target);

    float pixel_xcoord_normalized = (float)pixel_xcoord / (float)(width  - 1);
    float pixel_ycoord_normalized = (float)pixel_ycoord / (float)(height - 1);

    float h_coord_float = pixel_xcoord_normalized * (float)(h_size - 1);
    float h_val = (h[(int)floor(h_coord_float)] + h[(int)ceil(h_coord_float)]) * (float)0.5;

    float axis_ycoord_normalized = (float)1. - pixel_ycoord_normalized; // because image y-axis is upside-down
    float axis_ycoord = axis_ycoord_normalized * (axis_max - axis_min) + axis_min;
    float pixel_is_in_fluid = step(axis_ycoord, h_val);
    float3 color = pixel_is_in_fluid * (float3)(0., 0.5, 1.);

    convert_and_write_image(render_target, (int2)(pixel_xcoord, pixel_ycoord), color);
}
