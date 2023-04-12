/* These constants must be defined using the '-D' flag when compiling the kernels.
We rename them all here so that the IDE doesn't display "undeclared identifier" errors all over the rest of
the file, and so that their types are explicit.
*/
static constant const float  DT = TIME_STEP;
static constant const float2 SPATIAL_STEP = (float2)(SPATIAL_STEP_X, SPATIAL_STEP_Y);
static constant const int2   GRID_SIZE = (int2)(N_GRIDPOINTS_X, N_GRIDPOINTS_Y);

static constant const float3 FLUID_COLOR = (float3)(0.0f, 0.5f, 1.0f);
static constant const float3 BAD_COLOR   = (float3)(1.0f, 0.0f, 0.0f); // use to indicate errors

// acceleration due to gravity
#define G 9.81f

/*
Convert 2d logical index (x, y) in the grid to a 1d index for accessing the value at (x,y) in a buffer,
assuming the buffer data is in row-major order.
*/
// @todo should we start using 64-bit ints for linear indices to avoid overflow on large grids?
int gridindex_2d_to_1d(int2 ind) {
    return GRID_SIZE.x*ind.y + ind.x;
}

float choose_f(bool condition, float val_if_true, float val_if_false) {
    return condition*val_if_true + (!condition)*val_if_false;
}
float3 choose_f3(bool condition, float3 val_if_true, float3 val_if_false) {
    return condition*val_if_true + (!condition)*val_if_false;
}
int2 choose_i2(bool condition, int2 val_if_true, int2 val_if_false) {
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

kernel void iterate(global float* h, global float2* w) {
    /*
    Let
        rho = density
        h = fluid height or depth or whatever (distance between fluid surface and riverbed)
        u = component of fluid velocity in the x direction
        v = component of fluid velocity in the y direction
        g = acceleration due to gravity.
    Wikipedia gives the equations (I moved `rho` out of the derivatives, i.e. I've assumed incompressibility)
    1:
        rho dh/dt + rho d(hu)/dx + rho d(hv)/dy = 0
        rho d(hu)/dt + d/dx (rho h u^2 + 1/2 rho g h^2) + rho d(huv)/dy = 0
        rho d(hv)/dt + d/dy (rho h v^2 + 1/2 rho g h^2) + rho d(huv)/dx = 0.
    Factoring out rho,
    2:
        dh/dt + d(hu)/dx + d(hv)/dy = 0
        d(hu)/dt + d/dx (h u^2 + 1/2 g h^2) + d(huv)/dy = 0
        d(hv)/dt + d/dy (h v^2 + 1/2 g h^2) + d(huv)/dx = 0.
    Expanding the time-derivatives via the product rule,
    3:
        dh/dt + d(hu)/dx + d(hv)/dy = 0
        du/dt h + dh/dt u + d/dx (h u^2 + 1/2 g h^2) + d(huv)/dy = 0
        dv/dt h + dh/dt v + d/dy (h v^2 + 1/2 g h^2) + d(huv)/dx = 0.
    Rearranging,
    4:
        dh/dt = -[ d(hu)/dx + d(hv)/dy ]
        du/dt = -[ dh/dt u + d/dx (h u^2 + 1/2 g h^2) + d(huv)/dy ] / h
        dv/dt = -[ dh/dt v + d/dy (h v^2 + 1/2 g h^2) + d(huv)/dx ] / h.
    @note This is invalid for h=0, so we need to ensure that h=0 doesn't happen.
    Distributing the derivative over addition and constant multiplication,
    5:
        dh/dt = -[ d(hu)/dx + d(hv)/dy ]
        du/dt = -[ dh/dt u + d(h u^2)/dx + 1/2 g d(h^2)/dx + d(huv)/dy ] / h
        dv/dt = -[ dh/dt v + d(h v^2)/dy + 1/2 g d(h^2)/dy + d(huv)/dx ] / h.
    Now,
        let the vector w := (u,v);
        let `.^` be the component-wise exponent operation;
        let `flip((x,y))` := (y,x).
    Then (5) becomes
    6:
        dh/dt = -[ d(hu)/dx + d(hv)/dy ]
        dw/dt = -[ dh/dt w + grad(h w.^2) + 1/2 g grad(h^2) + flip(grad(huv)) ] / h.
    Expanding the first two `grad` terms,
    7:
        dh/dt = -[ d(hu)/dx + d(hv)/dy ]
        dw/dt = -[ dh/dt w + 2hw grad(w) + w.^2 grad(h) + gh grad(h) +  flip(grad(huv)) ] / h.

    We can compute all the spatial gradients using finite difference.
    Then we compute dh/dt, and use that to compute dw/dt.

    Boundary condition: reflective (i.e. wave just bounces off a wall)
    */

    // @todo some of the computations in this kernel are redundant, and some can be done at compile-time

    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int2 gid = (int2)(gid_x, gid_y);

    // l,r,t,b : left, right, top, bottom
    bool is_l_boundary = gid_x == 0;
    bool is_r_boundary = gid_x == GRID_SIZE.x-1;
    bool is_b_boundary = gid_y == 0;
    bool is_t_boundary = gid_y == GRID_SIZE.y-1;
    //
    bool is_lr_boundary = is_l_boundary || is_r_boundary;
    bool is_bt_boundary = is_b_boundary || is_t_boundary;

    /*
    The indices used for the finite difference gradient computation.
    In each dimension, we use centered difference unless the gridpoint is on the boundary in that dimension.
    */
    int ind_c = gridindex_2d_to_1d(gid); // c : "center"
    int ind_l = gridindex_2d_to_1d(choose_i2(is_l_boundary, gid, (int2)(gid.x-1, gid.y  )));
    int ind_r = gridindex_2d_to_1d(choose_i2(is_r_boundary, gid, (int2)(gid.x+1, gid.y  )));
    int ind_b = gridindex_2d_to_1d(choose_i2(is_b_boundary, gid, (int2)(gid.x  , gid.y-1)));
    int ind_t = gridindex_2d_to_1d(choose_i2(is_t_boundary, gid, (int2)(gid.x  , gid.y+1)));

    float  h_c = h[ind_c];
    float  h_l = h[ind_l];
    float  h_r = h[ind_r];
    float  h_b = h[ind_b];
    float  h_t = h[ind_t];
    //
    float2 w_c = w[ind_c];
    float2 w_l = w[ind_l];
    float2 w_r = w[ind_r];
    float2 w_b = w[ind_b];
    float2 w_t = w[ind_t];
    //
    // Multiply gradient computation by 0.5 if using centered difference.
    float2 grad_factor = (float2)(
        choose_f(is_lr_boundary, 1.f/SPATIAL_STEP.x, 0.5f/SPATIAL_STEP.x),
        choose_f(is_bt_boundary, 1.f/SPATIAL_STEP.y, 0.5f/SPATIAL_STEP.y)
    );

    float2 grad_h = (float2)(h_r   - h_l,   h_t   - h_b  ) * grad_factor;
    float2 grad_w = (float2)(w_r.x - w_l.x, w_t.y - w_b.y) * grad_factor;

    // dh/dt = -[ d(hu)/dx + d(hv)/dy ] = -[ grad(hw).x + grad(hw).y ]
    // where grad(hw) = w.*grad(h) + h*grad(w)
    float2 grad_hw = w_c*grad_h + h_c*grad_w;
    float dh_by_dt = -(grad_hw.x + grad_hw.y);

    // grad(huv) = uv grad(h) + h grad(uv) = uv grad(h) + h [u grad(v) + v grad(u)]
    // = uv grad(h) + h [u (0, grad(w).y) + v (grad(w).x, 0)]
    // = uv grad(h) + h [(u,v) .* (grad(w).y, grad(w).x)]
    // = w.x*w.y*grad(h) + h*w.*flip(grad(w)).
    float2 grad_huv = w_c.x*w_c.y*grad_h + h_c*grad_w.yx;
    float2 dw_by_dt = -( dh_by_dt*w_c + 2*h_c*w_c*grad_w + G*h_c*grad_h + grad_huv.yx ) / ((h_c > 0)*h_c + (h_c <= 0)*1.f);

    // @todo maybe some fancier time integration method (Runge-Kutta?)
    float  h_new = h_c + dh_by_dt*DT;
    float2 w_new = w_c + dw_by_dt*DT;
    // reflect if hitting boundaries
    w_new.x *= mul_if(is_lr_boundary, -1.f);
    w_new.y *= mul_if(is_bt_boundary, -1.f);

    barrier(CLK_GLOBAL_MEM_FENCE); // make sure all invocations have read from the buffers we're gonna modify
    h[ind_c] = h_new;
    w[ind_c] = w_new;
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
    pixel_ycoord_normalized = 1.f - pixel_ycoord_normalized; // because image y-axis is upside-down

    float h_xcoord_float = pixel_xcoord_normalized * (float)(GRID_SIZE.x - 1);
    float h_ycoord_float = pixel_ycoord_normalized * (float)(GRID_SIZE.y - 1);

    /* The sample point might fall between gridpoints; take the average value of the nearest gridpoints.
    Some or all of these gridpoints could be the same gridpoint, which should be fine.
    @note We use `min`/`max` here to prevent ouf-of-bounds accesses near boundary points caused by float
    imprecision. */
    int nearest_xcoord_left  = max((int)floor(h_xcoord_float), 0);
    int nearest_xcoord_right = min((int) ceil(h_xcoord_float), GRID_SIZE.x - 1);
    int nearest_ycoord_below = max((int)floor(h_ycoord_float), 0);
    int nearest_ycoord_above = min((int) ceil(h_ycoord_float), GRID_SIZE.y - 1);

    float h_val1 = h[gridindex_2d_to_1d((int2)(nearest_xcoord_left , nearest_ycoord_below))];
    float h_val2 = h[gridindex_2d_to_1d((int2)(nearest_xcoord_left , nearest_ycoord_above))];
    float h_val3 = h[gridindex_2d_to_1d((int2)(nearest_xcoord_right, nearest_ycoord_below))];
    float h_val4 = h[gridindex_2d_to_1d((int2)(nearest_xcoord_right, nearest_ycoord_above))];
    float h_val = (h_val1 + h_val2 + h_val3 + h_val4) * 0.25f;

    bool something_is_wrong = !isnormal(h_val); // +-zero, +-Inf, NaN, and subnormal values
    h_val = clamp(h_val, axis_min, axis_max);
    float brightness = (h_val - axis_min) / (axis_max - axis_min); // normalize to [0, 1]
    float3 color = choose_f3(something_is_wrong, BAD_COLOR, brightness * FLUID_COLOR);
    // if ((pixel_xcoord == width / 2) || (pixel_ycoord == height / 2)) color = BAD_COLOR;

    convert_and_write_image(render_target, (int2)(pixel_xcoord, pixel_ycoord), color);
}
