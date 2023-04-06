/* These constants must be defined using the '-D' flag when compiling the kernels.
We rename them all here so that the IDE doesn't display "undeclared identifier" errors all over the rest of
the file, and so that their types are explicit.
*/
static constant const float DT = TIME_STEP;
static constant const float DX = SPACIAL_STEP;
static constant const int   H_SIZE = N_GRIDPOINTS;

// acceleration due to gravity
#define G 9.81f

float choose(bool condition, float val_if_true, float val_if_false) {
    return condition*val_if_true + (!condition)*val_if_false;
}
/*
Returns `factor` if `condition`; otherwise just returns 1.
Use this for conditional multiplications, e.g.:
    float result = mul_if(i_want_to_mul_by_5, 5.f)*thing_to_multiply;
*/
float mul_if(bool condition, float factor) {
    return condition*factor + (float)(!condition);
}

kernel void iterate(global float* h, global float* v) {
    // @todo current version is numerically unstable
    /* Assuming an incompressible fluid (i.e. constant density) and 1 horizontal dimension:
    Let
        rho = density
        h = fluid height or depth or whatever (distance between fluid surface and riverbed)
        v = horizontal fluid velocity (mean across column, but we don't need to worry about that)
        g = acceleration due to gravity.
    Then
        rho dh/dt + rho d(hv)/dx = 0
        rho d(hv)/dt + d/dx (rho h v^2 + 1/2 rho g h^2) = 0.
    Expanding part of the latter equation using product rule,
        rho dh/dt + rho d(hv)/dx = 0
        rho (dh/dt v + dv/dt h) + d/dx (rho h v^2 + 1/2 rho g h^2) = 0.
    Rearranging,
        dh/dt = - d(hv)/dx
        dv/dt = (-d/dx (h v^2 + 1/2 g h^2) - dh/dt v) / h.
                        -----------------
                        let's call this term q (name chosen arbitrarily)
    @note This is invalid for h=0, so we need to ensure that h=0 doesn't happen.

    We can compute all the spatial gradients using finite difference.
    Then we compute dh/dt, and use that to compute dv/dt.

    Boundary condition: reflective (i.e. wave just bounces off a wall)
    */

    int gid = get_global_id(0);
    bool is_left_boundary  = gid == 0;
    bool is_right_boundary = gid == H_SIZE-1;
    bool is_boundary = is_left_boundary || is_right_boundary;
    bool is_interior = !is_boundary;

    /*
    The indices ("ind_forward" and "ind_backward") used for the finite difference gradient computation.
    We use centered difference on the interior, and forward or backward difference on the boundaries.
    */
    int ind_f = choose(is_right_boundary, gid, gid+1);
    int ind_b = choose(is_left_boundary , gid, gid-1);

    /*
        h_b : h_back,    i.e. h at previous grid point
        h_c : h_center,  i.e. h at this invocation's grid point
        h_f : h_forward, i.e. h at next grid point
    */
    float h_b = h[ind_b];
    float h_c = h[gid];
    float h_f = h[ind_f];
    float v_b = v[ind_b];
    float v_c = v[gid];
    float v_f = v[ind_f];

    // @todo some of these computations are redundant, and some can be done at compile-time
    float dhv_by_dx = (h_f*v_f - h_b*v_b) * mul_if(is_interior, 0.5f) / DX;
    float dq_by_dx = (
        ( h_f*(v_f*v_f) + 0.5*G*(h_f*h_f) )
      - ( h_b*(v_b*v_b) + 0.5*G*(h_b*h_b) )
    ) * mul_if(is_interior, 0.5f) / DX;
    float dh_by_dt = -dhv_by_dx;
    float dv_by_dt = (-dq_by_dx - dh_by_dt*v_c) / h_c;

    // @todo maybe some fancier time integration method (Runge-Kutta?)
    float h_new = h_c + dh_by_dt*DT;
    float v_new = v_c + dv_by_dt*DT;
    v_new *= mul_if(is_boundary, -1.f); // reflect

    barrier(CLK_GLOBAL_MEM_FENCE); // make sure all invocations have read from the buffers we're gonna modify
    h[gid] = h_new;
    v[gid] = v_new;
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
kernel void render(write_only image2d_t render_target, global float* h, float axis_min, float axis_max) {
    int pixel_xcoord = get_global_id(0);
    int pixel_ycoord = get_global_id(1);

    int width  = get_image_width(render_target);
    int height = get_image_height(render_target);

    float pixel_xcoord_normalized = (float)pixel_xcoord / (float)(width  - 1);
    float pixel_ycoord_normalized = (float)pixel_ycoord / (float)(height - 1);

    float h_coord_float = pixel_xcoord_normalized * (float)(H_SIZE - 1);
    float h_val = (h[(int)floor(h_coord_float)] + h[(int)ceil(h_coord_float)]) * 0.5f;

    float axis_ycoord_normalized = 1.f - pixel_ycoord_normalized; // because image y-axis is upside-down
    float axis_ycoord = axis_ycoord_normalized * (axis_max - axis_min) + axis_min;
    float pixel_is_in_fluid = step(axis_ycoord, h_val);
    float3 color = pixel_is_in_fluid * (float3)(0., 0.5, 1.);

    convert_and_write_image(render_target, (int2)(pixel_xcoord, pixel_ycoord), color);
}
